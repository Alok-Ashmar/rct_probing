import os
import json
from modules.ServerLogger import ServerLogger
from utils.redis_pool import get_redis

logger = ServerLogger()

probe_state_ttl = int(os.environ.get("REDIS_TTL_SECONDS_SESSION", 3600))


def probe_state_key(exp_id: str, su_id: str, qs_id: str, mo_id: str) -> str:
    return f"rct_probing:probe_state:{exp_id}:{su_id}:{qs_id}:{mo_id}"


def survey_details_key(exp_id: str, su_id: str, qs_id: str) -> str:
    return f"rct_probing:survey_details:{exp_id}:{su_id}:{qs_id}"


async def load_probe_state(key: str) -> dict:
    try:
        cached = await get_redis().get(key)
        if not cached:
            return {}
        return json.loads(cached)
    except Exception as exc:
        logger.error("Failed to load probe state from Redis")
        logger.error(exc)
        return {}


async def save_probe_state(key: str, state: dict) -> None:
    try:
        payload = json.dumps(state)
        redis = get_redis()
        if probe_state_ttl > 0:
            await redis.setex(key, probe_state_ttl, payload)
        else:
            await redis.set(key, payload)
    except Exception as exc:
        logger.error("Failed to save probe state to Redis")
        logger.error(exc)


async def load_survey_details(exp_id: str, su_id: str, qs_id: str) -> dict:
    try:
        raw = await get_redis().get(survey_details_key(exp_id, su_id, qs_id))
        return json.loads(raw) if raw else {}
    except Exception as exc:
        logger.error("Failed to load survey details from Redis")
        logger.error(exc)
        return {}


def build_probe_state(session_no: int, counter: int, ended: bool, simple_store: bool) -> dict:
    return {
        "session_no": session_no,
        "counter": counter,
        "ended": ended,
        "simple_store": simple_store,
    }


def apply_probe_state(probe, state: dict) -> None:
    if not state:
        return
    try:
        probe.counter = int(state.get("counter", probe.counter))
    except Exception:
        pass
    try:
        probe.ended = bool(state.get("ended", probe.ended))
    except Exception:
        pass
    try:
        probe.simple_store = bool(state.get("simple_store", probe.simple_store))
    except Exception:
        pass
