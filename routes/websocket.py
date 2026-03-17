import asyncio
from typing import Any, Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from models.Survey import Experiment, Question, Survey, UserResponse
from modules.ProdNSightGenerator import NSIGHT_v2
from modules.ProdProbe_v2 import Probe
from modules.ServerLogger import ServerLogger
from services.relevance_checker import RelevanceChecker
from services.repetition_checker import RepetitionChecker
from utils.db_extractor import cache_survey_details, db_extract_survey_details
from utils.redis_pool import get_redis
from utils.state_management import (
    load_probe_state,
    load_survey_details,
    probe_state_key,
    save_probe_state,
)

websocket_router = APIRouter(prefix="/ws", tags=["websocket", "ai-qa"])
logger = ServerLogger()

active_connections: Dict[str, WebSocket] = {}

NSIGHT_REQUIRED_FIELDS = {
    "quality",
    "relevance",
    "detail",
    "confusion",
    "negativity",
    "consistency",
    "confidence",
    "keywords",
    "reason",
    "gibberish_score",
}


def _normalize_text(value: str) -> str:
    return " ".join((value or "").split()).strip().casefold()


def _merge_metric(metric: dict[str, Any], value: Any) -> None:
    if isinstance(value, dict):
        metric.update(value)
    elif hasattr(value, "model_dump"):
        metric.update(value.model_dump())


def _has_complete_nsight(metric: dict[str, Any]) -> bool:
    return NSIGHT_REQUIRED_FIELDS.issubset(metric.keys())


async def _load_cached_payload(
    client_response: UserResponse,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    payload = await load_survey_details(
        str(client_response.exp_id),
        str(client_response.su_id),
        str(client_response.qs_id),
    )
    if payload:
        return payload, None

    payload, error = await asyncio.to_thread(db_extract_survey_details, client_response)
    if error or not payload:
        error_message = str(error or "Details not found!!")
        return None, {
            "error": True,
            "message": error_message,
            "code": 404,
        }

    await cache_survey_details(client_response, payload)
    return payload, None


@websocket_router.websocket("/ai-qa")
async def websocket_ai_qa(websocket: WebSocket):
    await websocket.accept()

    repetition_checker = RepetitionChecker(get_redis())

    try:
        while True:
            data = await websocket.receive_text()
            client_response = UserResponse.model_validate_json(data)

            payload, error = await _load_cached_payload(client_response)
            if error or not payload:
                await websocket.send_json(error or {
                    "error": True,
                    "message": "Details not found!!",
                    "code": 404,
                })
                continue

            experiment = Experiment.model_validate(payload.get("experiment") or {})
            survey = Survey.model_validate(payload.get("survey") or {})
            question = Question.model_validate(payload.get("question") or {})

            try:
                state_key = probe_state_key(
                    str(client_response.exp_id),
                    str(client_response.su_id),
                    str(client_response.qs_id),
                    str(client_response.mo_id),
                )
                probe_state = await load_probe_state(state_key)
                session_no = int(probe_state.get("session_no", 0))
                simple_store = bool(probe_state.get("simple_store", True))

                probe = Probe(
                    mo_id=client_response.mo_id,
                    metadata=survey,
                    question=question,
                    experiment=experiment,
                    simple_store=simple_store,
                    session_no=session_no,
                )
                probe.apply_state(probe_state)

                if _normalize_text(client_response.question) == _normalize_text(question.question):
                    probe.clear_memory()
                    session_no = probe.session_no + 1
                    probe = Probe(
                        mo_id=client_response.mo_id,
                        metadata=survey,
                        question=question,
                        experiment=experiment,
                        simple_store=simple_store,
                        session_no=session_no,
                    )
                    probe.apply_state(
                        {
                            "counter": 0,
                            "ended": False,
                            "simple_store": simple_store,
                        }
                    )

                await save_probe_state(state_key, probe.to_state())

                is_repetition = False
                if question.config.repetition:
                    is_repetition = await repetition_checker.question_check_repetition(client_response)

                if is_repetition:
                    repeated_response = {
                        "error": False,
                        "message": "streaming-started",
                        "code": 200,
                        "response": {
                            "question": "",
                            "min_probing": probe.question.config.min_probe,
                            "max_probing": probe.question.config.max_probe,
                            "is_repetition": True,
                        },
                    }
                    await websocket.send_json(repeated_response)
                    repeated_response["message"] = "streaming-ended"
                    await websocket.send_json(repeated_response)
                    continue

                stream, immediate_coro, detailed_coro = probe.gen_streamed_follow_up(
                    client_response.question,
                    client_response.response,
                )

                final_response = {
                    "error": False,
                    "message": "streaming-started",
                    "code": 200,
                    "response": {
                        "question": "",
                        "min_probing": probe.question.config.min_probe,
                        "max_probing": probe.question.config.max_probe,
                    },
                }

                immediate_task = asyncio.create_task(immediate_coro)
                detailed_task = asyncio.create_task(detailed_coro)
                queue: asyncio.Queue[Any | None] = asyncio.Queue()

                async def consume_stream():
                    try:
                        async for chunk in stream:
                            await queue.put(chunk)
                    except asyncio.CancelledError:
                        raise
                    finally:
                        await queue.put(None)

                stream_task = asyncio.create_task(consume_stream())

                metric: dict[str, Any] = {}
                try:
                    immediate_metric = await immediate_task
                    _merge_metric(metric, immediate_metric)

                    is_gibberish = (
                        metric.get("gibberish_score", 0) > probe.question.config.gibberish_score
                    )

                    if is_gibberish:
                        stream_task.cancel()
                        detailed_task.cancel()

                    RelevanceChecker.check_and_update_prompt(probe, metric)

                    final_response["response"] = {
                        **final_response["response"],
                        "ended": probe.ended,
                        "metrics": metric,
                        "is_gibberish": is_gibberish,
                        "is_repetition": False,
                    }
                    await websocket.send_json(final_response)

                    if not is_gibberish:
                        while True:
                            chunk = await queue.get()
                            if chunk is None:
                                break

                            if (
                                detailed_task.done()
                                and not detailed_task.cancelled()
                                and "quality" not in metric
                            ):
                                try:
                                    _merge_metric(metric, detailed_task.result())
                                    probe.ended = (
                                        metric.get("quality", 0)
                                        >= probe.question.config.quality_threshold
                                    )
                                    final_response["response"]["metrics"] = metric
                                    final_response["response"]["ended"] = probe.ended
                                except Exception as exc:
                                    logger.error(f"Error getting detailed metrics: {exc}")

                            final_response["message"] = "streaming"
                            final_response["response"]["question"] = (
                                chunk.content if hasattr(chunk, "content") else str(chunk)
                            )
                            await websocket.send_json(final_response)

                    if (
                        not is_gibberish
                        and detailed_task.done()
                        and not detailed_task.cancelled()
                        and "quality" not in metric
                    ):
                        _merge_metric(metric, detailed_task.result())
                        probe.ended = (
                            metric.get("quality", 0) >= probe.question.config.quality_threshold
                        )
                        final_response["response"]["metrics"] = metric
                        final_response["response"]["ended"] = probe.ended
                    elif (
                        not is_gibberish
                        and not detailed_task.done()
                        and not detailed_task.cancelled()
                    ):
                        detailed_metric = await detailed_task
                        _merge_metric(metric, detailed_metric)
                        probe.ended = (
                            metric.get("quality", 0) >= probe.question.config.quality_threshold
                        )
                        final_response["response"]["metrics"] = metric
                        final_response["response"]["ended"] = probe.ended

                    ended_response = {
                        **final_response,
                        "message": "streaming-ended",
                        "response": {
                            **final_response["response"],
                            "question": "",
                        },
                    }
                    await websocket.send_json(ended_response)
                    await save_probe_state(state_key, probe.to_state())

                    if probe.simple_store and _has_complete_nsight(metric):
                        nsight_v2 = NSIGHT_v2(
                            **metric,
                            question=client_response.question,
                            response=client_response.response,
                        )
                        probe.store_response(nsight_v2, probe.session_no)
                finally:
                    if not stream_task.done():
                        stream_task.cancel()
                    if not detailed_task.done() and not detailed_task.cancelled():
                        detailed_task.cancel()
                    await asyncio.gather(
                        stream_task,
                        detailed_task,
                        return_exceptions=True,
                    )

            except Exception as e:
                logger.error("Error in websocket AI QA:")
                logger.error(e)
                await websocket.send_json({
                    "error": True,
                    "message": str(e),
                    "code": 500,
                })

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error("WebSocket error:")
        logger.error(e)
        await websocket.close(code=1011, reason="Internal server error")
