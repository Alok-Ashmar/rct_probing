import os
import json
from redis import Redis
from modules.ServerLogger import ServerLogger

logger = ServerLogger()

redis_client = Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
probe_state_ttl = int(os.environ.get("REDIS_TTL_SECONDS_SESSION", 3600))


def probe_state_key(exp_id: str, su_id: str, qs_id: str, mo_id: str) -> str:
    return f"rct_probing:probe_state:{exp_id}:{su_id}:{qs_id}:{mo_id}"


def load_probe_state(key: str) -> dict:
    try:
        cached = redis_client.get(key)
        if not cached:
            return {}
        return json.loads(cached)
    except Exception as exc:
        logger.error("Failed to load probe state from Redis")
        logger.error(exc)
        return {}


def save_probe_state(key: str, state: dict) -> None:
    try:
        payload = json.dumps(state)
        if probe_state_ttl > 0:
            redis_client.setex(key, probe_state_ttl, payload)
        else:
            redis_client.set(key, payload)
    except Exception as exc:
        logger.error("Failed to save probe state to Redis")
        logger.error(exc)


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
