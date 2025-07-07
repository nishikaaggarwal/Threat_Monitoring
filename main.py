from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
from sklearn.ensemble import IsolationForest

app = FastAPI()

class UserActivity(BaseModel):
    user_id: str
    role: str
    resource: str
    action: str
    timestamp: str
    current_hour: int

class ActivityBatch(BaseModel):
    logs: List[UserActivity]

access_rules = {
    "analyst": ["read:/reports/conf"],
    "sysadmin": ["read:/etc/passwd", "write:/etc/passwd"],
    "intern": ["read:/training/data"]
}

user_logs = {
    "U001": ["read:/reports/conf"],
    "U002": ["read:/etc/passwd", "write:/etc/passwd"],
    "U003": ["read:/training/data"]
}

role_score = {"intern": 0, "analyst": 1, "sysadmin": 2}
resource_score = {
    "/training/data": 1,
    "/reports/conf": 2,
    "/secure/data": 3,
    "/etc/passwd": 3
}

@app.post("/analyze")
def check_logs(batch: ActivityBatch):
    all_results = []
    for log in batch.logs:
        full_action = f"{log.action}:{log.resource}"
        access_wrong = full_action not in access_rules.get(log.role, [])
        behavior_strange = full_action not in user_logs.get(log.user_id, [])
        hour = log.current_hour
        role_val = role_score.get(log.role, 0)
        resource_val = resource_score.get(log.resource, 1)
        df = pd.DataFrame([[hour, role_val, resource_val]], columns=["hour", "role", "resource"])
        model = IsolationForest(contamination=0.3, random_state=42)
        model.fit(df)
        prediction = model.predict(df)[0]
        is_anomaly = prediction == -1
        final_threat = access_wrong or behavior_strange or is_anomaly
        result = {
            "user_id": log.user_id,
            "role": log.role,
            "action": log.action,
            "resource": log.resource,
            "access_violation": access_wrong,
            "unusual_behavior": behavior_strange,
            "isf_anomaly": is_anomaly,
            "final_risk": final_threat
        }
        all_results.append(result)
    return all_results
