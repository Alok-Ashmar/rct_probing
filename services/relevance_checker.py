from langchain_core.messages import SystemMessage


class RelevanceChecker:
    @staticmethod
    def check_and_update_prompt(probe, metric):
        """Append the relevance recovery prompt when the score falls below threshold."""
        threshold = getattr(probe, "relevance_threshold", None)
        if threshold is None:
            return

        if isinstance(metric, dict):
            relevance = metric.get("relevance")
        else:
            relevance = getattr(metric, "relevance", None)

        if relevance is not None and relevance < threshold:
            RelevanceChecker.add_relevance_prompt(probe)

    @staticmethod
    def add_relevance_prompt(probe):
        if getattr(probe, "relevance_prompt_added", False):
            return

        probe.__system_prompt__ += f"\n{probe.__prompt_chunks__.get('relevance-chk', '')}"
        probe.relevance_prompt_added = True

        messages = probe._history.messages
        if messages and isinstance(messages[0], SystemMessage):
            messages[0] = SystemMessage(content=probe.__system_prompt__)
            probe._history.clear()
            for message in messages:
                probe._history.add_message(message)
