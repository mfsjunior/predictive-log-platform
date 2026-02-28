"""
Synthetic dataset generator for web log data.
Generates 5000 records with realistic distributions, intentional outliers,
correlation between hour and errors, and higher variance during peak hours.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_synthetic_dataset(n_records: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic web log dataset with realistic patterns.

    Args:
        n_records: Number of records to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: timestamp, method, path, status_code,
        response_time_ms, user_agent, ip_address, bytes_sent
    """
    rng = np.random.RandomState(seed)

    # ---------- Timestamps (last 30 days) ----------
    base_time = datetime(2025, 1, 1, 0, 0, 0)
    offsets_seconds = rng.randint(0, 30 * 24 * 3600, size=n_records)
    timestamps = [base_time + timedelta(seconds=int(s)) for s in sorted(offsets_seconds)]

    # ---------- HTTP Methods ----------
    methods = rng.choice(
        ["GET", "POST", "PUT", "DELETE", "PATCH"],
        size=n_records,
        p=[0.55, 0.25, 0.10, 0.05, 0.05],
    )

    # ---------- Paths ----------
    path_pool = [
        "/api/users", "/api/products", "/api/orders", "/api/auth/login",
        "/api/auth/logout", "/api/search", "/api/reports", "/api/health",
        "/api/dashboard", "/api/notifications", "/api/settings",
        "/api/upload", "/api/export", "/api/webhook",
    ]
    paths = rng.choice(path_pool, size=n_records)

    # ---------- Hours (extracted from timestamps) ----------
    hours = np.array([t.hour for t in timestamps])

    # ---------- Status Codes with hour-correlated error probability ----------
    status_codes = np.zeros(n_records, dtype=int)
    for i in range(n_records):
        h = hours[i]
        m = methods[i]

        # Base error probability
        base_error_prob = 0.08

        # Higher error rate during peak hours (9-12, 14-18)
        if 9 <= h <= 12 or 14 <= h <= 18:
            base_error_prob += 0.07
        # Even higher at lunch/evening transitions
        if h in (12, 13, 17, 18):
            base_error_prob += 0.05

        # POST/PUT/DELETE have slightly higher error rates
        if m in ("POST", "PUT", "DELETE"):
            base_error_prob += 0.03

        # Night hours have lower error rate
        if 0 <= h <= 5:
            base_error_prob -= 0.03

        base_error_prob = np.clip(base_error_prob, 0.02, 0.35)

        if rng.random() < base_error_prob:
            # Error response
            status_codes[i] = rng.choice(
                [400, 401, 403, 404, 500, 502, 503],
                p=[0.20, 0.10, 0.08, 0.30, 0.15, 0.07, 0.10],
            )
        else:
            # Success response
            status_codes[i] = rng.choice(
                [200, 201, 204, 301, 302],
                p=[0.65, 0.15, 0.10, 0.05, 0.05],
            )

    # ---------- Response Times with hour-dependent variance ----------
    response_times = np.zeros(n_records)
    for i in range(n_records):
        h = hours[i]
        sc = status_codes[i]

        # Base response time (ms)
        if sc >= 500:
            base_rt = 800.0
            std_rt = 400.0
        elif sc >= 400:
            base_rt = 150.0
            std_rt = 80.0
        else:
            base_rt = 200.0
            std_rt = 100.0

        # Peak hours increase response time
        if 9 <= h <= 12 or 14 <= h <= 18:
            base_rt *= 1.5
            std_rt *= 2.0

        # Very late night is faster
        if 0 <= h <= 5:
            base_rt *= 0.6
            std_rt *= 0.5

        response_times[i] = max(10.0, rng.normal(base_rt, std_rt))

    # ---------- Inject intentional outliers (2% of data) ----------
    n_outliers = int(n_records * 0.02)
    outlier_indices = rng.choice(n_records, size=n_outliers, replace=False)
    for idx in outlier_indices:
        # Extreme response times (3-15 seconds)
        response_times[idx] = rng.uniform(3000, 15000)
        # Some extreme outliers also get error codes
        if rng.random() < 0.6:
            status_codes[idx] = rng.choice([500, 502, 503])

    # ---------- User Agents ----------
    user_agent_pool = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2) Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "PostmanRuntime/7.36.0",
        "python-requests/2.31.0",
        "curl/8.4.0",
        "Apache-HttpClient/4.5.14 (Java/21)",
        "okhttp/4.12.0",
    ]
    user_agents = rng.choice(user_agent_pool, size=n_records)

    # ---------- IP Addresses ----------
    ips = [
        f"{rng.randint(10, 200)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
        for _ in range(n_records)
    ]

    # ---------- Bytes Sent ----------
    bytes_sent = rng.randint(100, 50000, size=n_records)
    # Error responses send fewer bytes
    bytes_sent[status_codes >= 400] = rng.randint(50, 500, size=np.sum(status_codes >= 400))

    df = pd.DataFrame({
        "timestamp": timestamps,
        "method": methods,
        "path": paths,
        "status_code": status_codes,
        "response_time_ms": np.round(response_times, 2),
        "user_agent": user_agents,
        "ip_address": ips,
        "bytes_sent": bytes_sent,
    })

    return df


def save_dataset(df: pd.DataFrame, filepath: str) -> str:
    """Save dataset to CSV file."""
    df.to_csv(filepath, index=False)
    return filepath


if __name__ == "__main__":
    dataset = generate_synthetic_dataset()
    save_dataset(dataset, "./data/web_logs.csv")
    print(f"Generated {len(dataset)} records")
    print(f"Error rate: {(dataset['status_code'] >= 400).mean():.2%}")
    print(f"Avg response time: {dataset['response_time_ms'].mean():.2f} ms")
    print(dataset.describe())
