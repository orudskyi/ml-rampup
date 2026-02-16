import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# 1. Data Generation
N = 1_000_000
rng = np.random.default_rng(seed=42)

endpoints_list = [
    "/api/v1/login",
    "/api/v1/pay",
    "/home",
    "/dashboard",
    "/api/v1/logout",
]
status_codes_list = [200, 201, 301, 400, 404, 500, 502]
status_probs = [0.7, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02]

print("Generating synthetic log data...")

start_date = datetime.now()
random_seconds = rng.integers(0, 30 * 24 * 3600, size=N)
timestamps = start_date - pd.to_timedelta(random_seconds, unit="s")

# Endpoint, Status, and Latency generation
endpoints = rng.choice(endpoints_list, size=N)
status_codes = rng.choice(status_codes_list, size=N, p=status_probs)
latency = rng.integers(10, 5001, size=N)  # 10ms to 5000ms

# Create Initial DataFrame
df = pd.DataFrame(
    {
        "timestamp": timestamps,
        "endpoint": endpoints,
        "status_code": status_codes,
        "latency_ms": latency,
    }
)

# Converting string/int objects to 'category' drastically reduces memory usage
df["endpoint"] = df["endpoint"].astype("category")
df["status_code"] = df["status_code"].astype("category")
df["timestamp"] = pd.to_datetime(df["timestamp"])

print(f"Data generated successfully. Shape: {df.shape}")
print("-" * 40)
# Display memory usage to verify optimization
df.info()
print("-" * 40)


# 2. ANALYTICS

# Calculate 95th percentile (P95) latency per endpoint
p95_per_endpoint = df.groupby("endpoint", observed=True)["latency_ms"].quantile(0.95)

print("\n--- P95 Latency per Endpoint ---")
print(p95_per_endpoint)


# Count errors (status >= 400) per hour using Resampling
hourly_errors = (
    df[df["status_code"].astype(int) >= 400]
    .resample("h", on="timestamp")  # Працюємо без set_index
    .size()
)

print("\n--- Errors per Hour (Top 5 busiest hours) ---")
print(hourly_errors.sort_values(ascending=False).head(5))

mean_latency_by_group = df.groupby("endpoint", observed=True)["latency_ms"].transform(
    "mean"
)

df["is_slow"] = df["latency_ms"] > mean_latency_by_group

print("\n--- Result with 'is_slow' column ---")
print(df[["endpoint", "latency_ms", "is_slow"]].head())

slow_count = df["is_slow"].sum()
print(f"Total slow requests flagged: {slow_count}")
