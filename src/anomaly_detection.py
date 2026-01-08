import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import time

# -----------------------------
# Generate Simulated Market Data
# -----------------------------
np.random.seed(42)
data_size = 1000

price = np.cumsum(np.random.normal(0, 1, data_size)) + 100
volume = np.random.normal(1000, 100, data_size)

# Inject anomalies
price[950:960] += 25
volume[970:980] += 800

df = pd.DataFrame({
    "price": price,
    "volume": volume
})

# -----------------------------
# Feature Engineering
# -----------------------------
df["price_change"] = df["price"].pct_change().fillna(0)
df["volume_change"] = df["volume"].pct_change().fillna(0)

df["price_mean"] = df["price"].rolling(10).mean().fillna(method="bfill")
df["price_std"] = df["price"].rolling(10).std().fillna(method="bfill")

features = df[
    ["price_change", "volume_change", "price_mean", "price_std"]
]

# -----------------------------
# Train Isolation Forest Model
# -----------------------------
model = IsolationForest(
    n_estimators=200,
    contamination=0.03,
    random_state=42
)

model.fit(features)

df["anomaly_score"] = model.decision_function(features)
df["anomaly"] = model.predict(features)

# -----------------------------
# Alert Prioritization
# -----------------------------
def alert_level(score):
    if score < -0.15:
        return "HIGH"
    elif score < -0.05:
        return "MEDIUM"
    else:
        return "LOW"

df["alert_level"] = df["anomaly_score"].apply(alert_level)

# -----------------------------
# Display Detected Anomalies
# -----------------------------
alerts = df[df["anomaly"] == -1][
    ["price", "volume", "anomaly_score", "alert_level"]
]

print("Detected Anomalies:")
print(alerts.head(10))

# -----------------------------
# Live / Near Real-Time Detection
# -----------------------------
print("\nLive Anomaly Detection:\n")

for i in range(990, 1000):
    row = features.iloc[i].values.reshape(1, -1)
    score = model.decision_function(row)[0]
    level = alert_level(score)

    print(f"Time {i} | Score: {score:.3f} | Alert Level: {level}")
    time.sleep(1)
