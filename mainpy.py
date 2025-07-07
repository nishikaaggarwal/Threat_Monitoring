import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

random_seed = np.random.RandomState(12)

x_train = 0.5 * random_seed.randn(500, 2)
x_train = np.r_[x_train + 3, x_train]
x_train = pd.DataFrame(x_train, columns=["x", "y"])

x_test = 0.5 * random_seed.randn(500, 2)
x_test = np.r_[x_test + 3, x_test]
x_test = pd.DataFrame(x_test, columns=["x", "y"])

x_outlier = random_seed.uniform(low=-5, high=5, size=(50, 2))
x_outlier = pd.DataFrame(x_outlier, columns=["x", "y"])

plt.scatter(x_train.x, x_train.y, c="white", s=50, edgecolor="black")
plt.scatter(x_test.x, x_test.y, c="green", s=50, edgecolor="black")
plt.scatter(x_outlier.x, x_outlier.y, c="blue", s=50, edgecolor="black")
plt.xlim((-6, 6))
plt.ylim((-6, 6))
plt.legend(["train", "test", "outliers"], loc="lower right")
plt.show()

X_combined = pd.concat([x_test, x_outlier], ignore_index=True)
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(x_train)
predictions = model.predict(X_combined)

plt.figure(figsize=(8, 6))
for i, pred in enumerate(predictions):
    color = 'green' if pred == 1 else 'red'
    plt.scatter(X_combined.iloc[i, 0], X_combined.iloc[i, 1], c=color, s=50, edgecolor="black")
plt.title("Anomaly Detection")
plt.xlabel("X")
plt.ylabel("Y")
plt.xlim((-6, 6))
plt.ylim((-6, 6))
plt.grid(True)
plt.show()

data = [
    ["U001", "analyst", "/reports/conf", "read", "2025-07-07 12:34:00", 9, 12, "10.0.0.5"],
    ["U002", "sysadmin", "/etc/passwd", "write", "2025-07-07 03:22:00", 9, 3, "192.168.1.100"],
    ["U003", "intern", "/secure/data", "download", "2025-07-07 23:45:00", 9, 23, "10.0.0.15"]
]
columns = ["user_id", "role", "resource", "action", "timestamp", "usual_start", "current_hour", "ip_address"]
df = pd.DataFrame(data, columns=columns)
df["timestamp"] = pd.to_datetime(df["timestamp"])

ACCESS_RULES = {
    "analyst": ["read:/reports/conf"],
    "sysadmin": ["read:/etc/passwd", "write:/etc/passwd", "restart:/system"],
    "intern": ["read:/training/data"]
}
def is_action_allowed(role, resource, action):
    return f"{action}:{resource}" in ACCESS_RULES.get(role, [])

df["access_violation"] = df.apply(lambda row: not is_action_allowed(row["role"], row["resource"], row["action"]), axis=1)

user_history = {
    "U001": ["read:/reports/conf"],
    "U002": ["read:/etc/passwd", "write:/etc/passwd"],
    "U003": ["read:/training/data"]
}
def is_behavior_unusual(user_id, resource, action):
    return f"{action}:{resource}" not in user_history.get(user_id, [])

df["unusual_behavior"] = df.apply(lambda row: is_behavior_unusual(row["user_id"], row["resource"], row["action"]), axis=1)
df["threat_flag"] = df.apply(lambda row: row["access_violation"] or row["unusual_behavior"], axis=1)

role_map = {"intern": 0, "analyst": 1, "sysadmin": 2}
df["role_level"] = df["role"].map(role_map)

resource_sensitivity = {
    "/training/data": 1,
    "/reports/conf": 2,
    "/secure/data": 3,
    "/etc/passwd": 3
}
df["resource_level"] = df["resource"].map(resource_sensitivity)

features = df[["current_hour", "role_level", "resource_level"]]
model = IsolationForest(contamination=0.3, random_state=42)
model.fit(features)
df["isf_prediction"] = model.predict(features)
df["isf_anomaly"] = df["isf_prediction"] == -1
df["final_risk"] = df.apply(lambda row: row["threat_flag"] or row["isf_anomaly"], axis=1)

print(df[["user_id", "final_risk"]])
