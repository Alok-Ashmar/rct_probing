from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict
from .payload import LLMEnum, PyObjectId

class status(str, Enum):
    active = "active"
    draft = "draft"


class ExperimentConfig(BaseModel):
    add_context: bool = True


class QuestionConfig(BaseModel):
    min_probe: int = 1
    max_probe: int = 2
    add_context: bool = True
    allow_pasting: bool = False
    quality_threshold: int = 4
    gibberish_score: int = 4
    repetition: Optional[bool] = True
    relevance_threshold: Optional[int] = 4


class Experiment(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    exp_id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    title: str = Field(default_factory="", alias="experiment_name")
    experiment_description: str = ""
    experiment_status: Optional[status]
    config: ExperimentConfig = Field(default_factory=ExperimentConfig, alias="exp_flags")


class Survey(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    su_id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    language: str = "English"


class Question(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    qs_id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    question: str = Field(default_factory="", alias="qs_question")
    question_intent: str = ""
    config: QuestionConfig = Field(default_factory=QuestionConfig, alias="qs_flags")


class UserResponse(BaseModel):
    exp_id: str
    su_id: str
    mo_id: str
    qs_id: str
    asset_id: str
    question: str
    response: str
