import json
from typing import Dict
from models.Survey import UserResponse, Survey, Question, Experiment
from modules.ServerLogger import ServerLogger
from modules.ProdProbe_v2 import Probe, NSIGHT_v2
from utils.db_extractor import db_extract_survey_details
from utils.state_management import (
    redis_client,
    probe_state_key,
    load_probe_state,
    save_probe_state,
)
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

websocket_router = APIRouter(prefix="/ws", tags=["websocket", "ai-qa"])
logger = ServerLogger()


active_connections: Dict[str, WebSocket] = {}

def _normalize_text(value: str) -> str:
    return " ".join((value or "").split()).strip().casefold()

@websocket_router.websocket("/ai-qa")
async def websocket_ai_qa(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_text()
            client_response = UserResponse.model_validate_json(data)
            
            redis_key = f"rct_probing:survey_details:{client_response.exp_id}:{client_response.su_id}:{client_response.qs_id}"
            cached_payload = redis_client.get(redis_key)
            if not cached_payload:
                payload, error = db_extract_survey_details(
                    client_response=client_response,
                    redis_client=redis_client
                )
                if error or not payload:
                    await websocket.send_json(error or {
                        "error": True,
                        "message": "Details not found!!",
                        "code": 404
                    })
                    continue
            else:
                payload = json.loads(cached_payload)

            experiment = Experiment.model_validate(payload.get("experiment") or {})
            survey = Survey.model_validate(payload.get("survey") or {})
            question = Question.model_validate(payload.get("question") or {})

            try:
                probe = None
                state_key = probe_state_key(str(client_response.exp_id), str(client_response.su_id), str(client_response.qs_id), str(client_response.mo_id))
                probe_state = load_probe_state(state_key)
                session_no = int(probe_state.get("session_no", 0))
                simple_store = bool(probe_state.get("simple_store", True))

                probe = Probe(
                    mo_id=client_response.mo_id, 
                    metadata=survey, 
                    question=question, 
                    experiment=experiment, 
                    simple_store=simple_store, 
                    session_no=session_no
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
                        session_no=session_no
                    )
                    probe.apply_state(
                        {
                            "counter": 0,
                            "ended": False,
                            "simple_store": simple_store
                        }
                    )
                save_probe_state(state_key, probe.to_state())

                # Generate follow-up using the probe
                stream, metric_stream = probe.gen_streamed_follow_up(client_response.question, client_response.response)
                final_response = {
                    "error": False,
                    "message": "streaming-started",
                    "code": 200,
                    "response": {
                        "question": "",
                        "min_probing": probe.question.config.min_probe,
                        "max_probing": probe.question.config.max_probe,
                    }
                }
                
                ended_response = {}

                async for metric in metric_stream:
                    final_response["message"] = "streaming-started"
                    final_response["response"] = {
                        **final_response["response"],
                        "ended": True if metric.quality >= probe.question.config.quality_threshold else False,
                        "metrics": metric.model_dump(),
                        "is_gibberish": True if metric.gibberish_score > question.config.gibberish_score else False
                    }
                    ended_response = final_response.copy()
                    ended_response["message"] = "streaming-ended"
                    await websocket.send_json(final_response)

                if final_response["response"]["is_gibberish"] == False:
                    async for chunk in stream:
                        final_response["message"] = "streaming"
                        final_response["response"] = {
                            **final_response["response"],
                            "question": chunk.content,
                            "ended": probe.ended,
                        }
                        await websocket.send_json(final_response)

                await websocket.send_json(ended_response)
                save_probe_state(state_key, probe.to_state())
                
                if probe.simple_store:
                    nsight_v2 = NSIGHT_v2(**{**metric.model_dump(), "question": client_response.question, "response": client_response.response})
                    probe.store_response(nsight_v2, probe.session_no)

            except Exception as e:
                logger.error("Error in websocket AI QA:")
                logger.error(e)
                await websocket.send_json({
                    "error": True,
                    "message": str(e),
                    "code": 500
                })

    except WebSocketDisconnect:
        logger.info(f"Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error:")
        logger.error(e)
        await websocket.close(code=1011, reason="Internal server error")
