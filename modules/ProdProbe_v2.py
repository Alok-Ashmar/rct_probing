import os
import pytz
import warnings
from redis import Redis
from bson import ObjectId
from datetime import datetime
from langsmith import traceable
from typing import AsyncIterable
from modules.LLMAdapter import LLMAdapter
from modules.MongoWrapper import monet_db
from modules.ServerLogger import ServerLogger
from langchain_core.messages import SystemMessage
from models.Survey import Survey, Question, Experiment
from modules.ProdNSightGenerator import NSIGHT, NSIGHT_v2
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.chat_message_histories import RedisChatMessageHistory

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pydantic.*"
)

india = pytz.timezone('Asia/Kolkata')
logger = ServerLogger()

rct_probe_responses = monet_db.get_collection("rct_probe_responses")

class Probe(LLMAdapter):

    __version__ = "3.0.0"

    invalid = False

    def __init__(self,
        mo_id: str,
        metadata: Survey,
        question: Question,
        experiment: Experiment,
        simple_store=False,
        session_no:int = 0,
        ):
        super().__init__("chatgpt", 0.7, streaming=True)
        self.counter = 0
        self.ended = False
        self.mo_id = mo_id
        self.metadata = metadata
        self.question = question
        self.experiment = experiment
        self.session_no = session_no
        self.simple_store = simple_store
        self.su_id = self.metadata.su_id
        self.qs_id = self.question.qs_id
        self.__metric_llm__ = self.llm.with_structured_output(NSIGHT)
        self.id = f"{self.metadata.su_id}:{self.question.qs_id}:{mo_id}"

        if self.question.config.min_probe > self.question.config.max_probe:
            self.invalid = True

        self._history_redis_url = os.environ.get(
            "REDIS_URL",
            "redis://localhost:6379/0"
        )
        self._redis = Redis.from_url(self._history_redis_url)

        self.__prompt_chunks__ = {
            "main-chk": """
                You are a video analysis partner. Your goal is to extract truth from the user's input based *strictly* on the provided context description, while **mirroring the user's level of specificity**.
                    1. **Valid/General Subject:** If the user uses general terms (e.g., "the actor", "the music"), ask a detail question using those SAME general terms. **DO NOT** insert specific character/actor names from context unless the user wrote them first.
                    2. **Mixed Subject (Real + Fake):** If the user links an unverified subject with a verified one, **IGNORE the unverified subject** and ask only about the verified one.
                    3. **Purely Fake/Off-Topic:** If the user *only* mentions a person/object NOT in the context, you MUST ask: "Where did you notice [Subject] in **[Video Title]**?".""",

            "rule-chk": """
                Subject Verification Logic (MANDATORY)
                    - **Scenario A: Purely On-Topic / General**
                    (User mentions verified elements or general concepts like 'the actor', 'the setting')
                    -> Action: Ask a natural follow-up about visual/audio details.
                    -> **CRITICAL:** Use generic terms (e.g., "the performer", "that character"). **NEVER** swap a general word for a specific name (e.g., do NOT say the Actor's Name) unless the user named them first.

                    - **Scenario B: Mixed Input (Verified + Unverified)**
                    (User links a correct element with an incorrect/hallucinated one)
                    -> Action: The user is adding false details. **Pivot immediately to the Verified element.**
                    -> Example: "What specific details of **[Verified Subject's]** performance stood out in that moment?" (Ignore the incorrect part).

                    - **Scenario C: Purely Off-Topic**
                    (User mentions a person/object completely absent from the context)
                    -> Action: Challenge them using the Video Title from the context.
                    -> Example: "Where did you notice [Unverified Subject] *in [Video Title]*?"

                Stay Anchored
                    - Keep the conversation relevant to the original question.
                    - Do not introduce new themes or interpretations.
                    - If on topic, use user's last idea to build your follow-up.
                    - Redirect to original question's intent/context if user diverts too much away from the topic.
                    - Do not validate hallucinations.

                Ask One Clear Question
                    - Keep it short (max 15 words).
                    - No multi-part questions.
                    - Avoid emotional or symbolic language unless the user introduces it.

                Encourage Elaboration
                    - Provide hints and contexts subtly wherever required.
                    - Focus on visible evidence.
            """
        }

        self.__system_prompt__ = PromptTemplate(
            template = """
                {main-chk}
                {rule-chk}
                """
            ).invoke(
                {
                    "main-chk": self.__prompt_chunks__["main-chk"],
                    "rule-chk": self.__prompt_chunks__["rule-chk"]
                }
            ).text
    
        # survey level context (switch)
        if self.experiment.config.add_context:
            self.__system_prompt__ = PromptTemplate(
                template = """
                    {main_prompt}
                    
                    Survey Description: {survey_description}
                    
                """
            ).invoke(
                {
                    "main_prompt": self.__system_prompt__,
                    "survey_description": self.experiment.experiment_description
                }
            ).text

        # survey question level context (switch) 
        if self.question.config.add_context:
            self.__system_prompt__ = PromptTemplate(
                template = """
                    {main_prompt}
                    
                    Original Question: {original_question}
                    
                    Question Intended Purpose: {question_intent}
                """
            ).invoke(
                {
                    "main_prompt": self.__system_prompt__,
                    "original_question": self.question.question,
                    "question_intent": self.question.question_intent
                }
            ).text        
        else:
            self.__system_prompt__ = PromptTemplate(
                template = """
                    {main_prompt}
                    
                    Original Question: {original_question}
                """
            ).invoke(
                {
                    "main_prompt": self.__system_prompt__,
                    "original_question": self.question.question
                }
            ).text
            
        # language prompt (switch)
        if self.metadata.language != "English":
            self.__system_prompt__ = PromptTemplate(
                template = """
                    {main_prompt}

                    <-- Language Instruction -->
                    Please ask Questions in {language} language.
                    <-- Language Instruction -->
                """
            ).invoke(
                {
                    "main_prompt": self.__system_prompt__,
                    "language": self.metadata.language
                }
            ).text
        
        self._history = RedisChatMessageHistory(
            session_id=self._session_id(),
            url=self._history_redis_url,
            ttl=int(os.environ.get("REDIS_TTL_SECONDS", 3600))
        )

        self._ensure_system_message()


    def _session_id(self) -> str:
        return f"rct_probing:{self.id}:{self.session_no}"


    def _ensure_system_message(self):
        if not self._history.messages:
            self._history.add_message(SystemMessage(content=self.__system_prompt__))

    def to_state(self) -> dict:
        return {
            "session_no": self.session_no,
            "counter": self.counter,
            "ended": self.ended,
            "simple_store": self.simple_store,
        }

    def apply_state(self, state: dict):
        if not state:
            return
        try:
            self.counter = int(state.get("counter", self.counter))
        except Exception:
            pass
        try:
            self.ended = bool(state.get("ended", self.ended))
        except Exception:
            pass
        try:
            self.simple_store = bool(state.get("simple_store", self.simple_store))
        except Exception:
            pass

    def clear_memory(self):
        try:
            self._history.clear()
        except Exception as e:
            logger.error("Failed to clear Redis chat history")
            logger.error(e)


    async def _stream_with_history_update(self, chain, inputs: dict):
        full_content = ""
        async for chunk in chain.astream(inputs):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            full_content += content
            yield chunk
        if full_content:
            self._history.add_ai_message(full_content)


    def gen_streamed_follow_up(self, question: str, response: str) -> tuple[AsyncIterable[str], AsyncIterable[NSIGHT]]:
        next_counter = self.counter + 1
        user_text = f"Response {next_counter}. {response}"
        self._history.add_user_message(user_text)
        self.counter = next_counter
        prompt = ChatPromptTemplate.from_messages(self._history.messages)
        chain = prompt | self.llm
        metric_chain = prompt | self.__metric_llm__

        llm_stream: str = self._stream_with_history_update(chain, {})
        
        metric_llm_stream: NSIGHT = metric_chain.astream({})
        return (llm_stream, metric_llm_stream)


    def store_response(self, nsight_v2: NSIGHT_v2, session_no: int):
        now_india = datetime.now(india)
        insert_one_res = rct_probe_responses.insert_one({
            **nsight_v2.model_dump(),
            "ended": self.ended,
            "exp_id": self.experiment.exp_id,
            "mo_id": self.mo_id,
            "su_id": self.su_id, 
            "qs_id": self.qs_id,
            "qs_no": self.counter + 1,
            "created_at": now_india.isoformat(),
            "session_no": session_no,
        })
        logger.info("Inserted one doc successfully")
        logger.info(insert_one_res)
        return insert_one_res
