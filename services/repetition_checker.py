import json
import re
from models.Survey import UserResponse
from modules.ServerLogger import ServerLogger

logger = ServerLogger()


class RepetitionChecker:
    """Check whether the user's current answer already exists in Redis history."""

    def __init__(self, redis_client):
        self.redis_client = redis_client

    async def _read_key(self, key: bytes | str) -> list | dict | None:
        key_type_raw = await self.redis_client.type(key)
        key_type = key_type_raw.decode("utf-8") if isinstance(key_type_raw, bytes) else str(key_type_raw)

        if key_type == "string":
            raw = await self.redis_client.get(key)
            return json.loads(raw) if raw else None

        if key_type == "list":
            items = await self.redis_client.lrange(key, 0, -1)
            result = []
            for item in items:
                try:
                    result.append(json.loads(item))
                except Exception:
                    result.append(item.decode("utf-8") if isinstance(item, bytes) else item)
            return result

        if key_type == "hash":
            raw = await self.redis_client.hgetall(key)
            return {
                (k.decode("utf-8") if isinstance(k, bytes) else k): (
                    json.loads(v) if v else None
                )
                for k, v in raw.items()
            }

        if key_type == "set":
            items = await self.redis_client.smembers(key)
            return [json.loads(item) if item else None for item in items]

        logger.error(f"Unsupported Redis key type '{key_type}' for key: {key}")
        return None

    @staticmethod
    def _trailing_index(key: bytes | str) -> int:
        try:
            key_text = key.decode("utf-8") if isinstance(key, bytes) else str(key)
            return int(key_text.rsplit(":", 1)[-1])
        except (ValueError, AttributeError):
            return -1

    @staticmethod
    def _extract_content(raw: str) -> str:
        return re.sub(r"^Response\s+\d+\.\s*", "", raw).strip()

    async def _check_repetition(self, survey_response: UserResponse, pattern: str) -> bool:
        try:
            matched_keys = []
            async for key in self.redis_client.scan_iter(match=pattern):
                matched_keys.append(key)
        except Exception as exc:
            logger.error(f"Failed to scan Redis keys for pattern: {pattern}")
            logger.error(exc)
            return False

        if not matched_keys:
            return False

        latest_key = max(matched_keys, key=self._trailing_index)

        try:
            messages = await self._read_key(latest_key)
        except Exception as exc:
            logger.error(f"Failed to read Redis key: {latest_key}")
            logger.error(exc)
            return False

        human_messages = [
            message
            for message in (messages or [])
            if isinstance(message, dict) and message.get("type") == "human"
        ]

        past_responses = [
            self._extract_content(message["data"]["content"])
            for message in human_messages
            if message.get("data", {}).get("content")
        ]

        return survey_response.response in past_responses

    async def survey_check_repetition(self, survey_response: UserResponse) -> bool:
        pattern = f"message_store:rct_probing:{survey_response.su_id}:*:{survey_response.mo_id}:*"
        return await self._check_repetition(survey_response, pattern)

    async def question_check_repetition(self, survey_response: UserResponse) -> bool:
        pattern = (
            f"message_store:rct_probing:{survey_response.su_id}:"
            f"{survey_response.qs_id}:{survey_response.mo_id}:*"
        )
        return await self._check_repetition(survey_response, pattern)
