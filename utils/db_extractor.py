import os
import json
from typing import Any
from redis import Redis
from bson import ObjectId
from models.Survey import UserResponse
from modules.MongoWrapper import monet_db
from models.Survey import Survey, Experiment, Question

experiment_collection = monet_db.get_collection("experiments")
survey_collection = monet_db.get_collection("surveys")
question_collection = monet_db.get_collection("surveys")

def db_extract_survey_details(
    client_response: UserResponse, redis_client: Redis
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        experiment_details = experiment_collection.find_one(
            {
                "_id": ObjectId(client_response.exp_id)
            }
        )
        if not experiment_details:
            return None, "Experiment not found"

        # Extract survey details using survey ID
        survey_pipeline = [
            {
                "$match": {
                    "exp_id": ObjectId(client_response.exp_id),
                }
            },
            {
                "$unwind": "$survey",
            },
            {
                "$match": {
                    "survey._id": ObjectId(client_response.su_id),
                }
            },
            {
                "$project": {
                    "_id": "$survey._id",
                    "type": "$survey.type",
                }
            },
        ]
        survey_details = next(survey_collection.aggregate(survey_pipeline), None)
        if not survey_details:
            return None, "Survey not found"

        # Extract question details using question ID
        question_pipeline = [
            {
                "$match": {
                    "exp_id": ObjectId(client_response.exp_id),
                }
            },
            {
                "$unwind": "$survey",
            },
            {
                "$match": {
                    "survey._id": ObjectId(client_response.su_id),
                }
            },
            {
                "$unwind": "$survey.questions",
            },
            {
                "$match": {
                    "survey.questions._id": ObjectId(client_response.qs_id),
                }
            },
            {
                "$replaceRoot": {
                    "newRoot": "$survey.questions",
                }
            },
        ]
        question_details = next(question_collection.aggregate(question_pipeline), None)
        if not question_details:
            return None, "Question not found"


        experiment = Experiment(**experiment_details)
        survey = Survey(**survey_details)
        question = Question(**question_details)

        payload = {
            "experiment": experiment.model_dump(mode="json"),
            "survey": survey.model_dump(mode="json"),
            "question": question.model_dump(mode="json"),
        }

        redis_key = f"rct_probing:survey_details:{client_response.exp_id}:{client_response.su_id}:{client_response.qs_id}"
        redis_ttl = int(os.environ.get("REDIS_TTL_SECONDS_SURVEY", 86400))
        redis_payload = json.dumps(payload)
        if redis_ttl > 0:
            redis_client.setex(redis_key, redis_ttl, redis_payload)
        else:
            redis_client.set(redis_key, redis_payload)

        return payload, None
    except Exception as exc:
        return None, str(exc)
    