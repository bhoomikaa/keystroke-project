# Build feature dataset from ALL keystroke files
# One row per file (one user-session sample)

import os
import numpy as np
import pandas as pd
from collections import defaultdict

DATA_ROOT = "../data/UB_keystroke_dataset"
OUTPUT_CSV = "../outputs/keystroke_features.csv"


def extract_features(file_path):
    events = []

    # ----------------------------
    # Step 1: Parse events
    # ----------------------------
    with open(file_path, "r", encoding="latin1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) >= 3 and parts[-1].isdigit():
                key = " ".join(parts[:-2])
                event_type = parts[-2]
                timestamp = int(parts[-1])
                events.append((timestamp, event_type, key))

    if len(events) == 0:
        return None

    events.sort(key=lambda x: x[0])

    # ----------------------------
    # Step 2: Hold times
    # ----------------------------
    keydown_times = defaultdict(list)
    hold_times = []
    per_key_holds = defaultdict(list)
    per_key_inter = defaultdict(list)

    for ts, ev, key in events:
        if ev == "KeyDown":
            keydown_times[key].append(ts)
        elif ev == "KeyUp":
            if keydown_times[key]:
                down_ts = keydown_times[key].pop(0)
                duration = ts - down_ts
                hold_times.append(duration)
                per_key_holds[key].append(duration)

    # ----------------------------
    # Step 3: Inter-key delays
    # ----------------------------
    inter_key_delays = []
    last_keydown_ts = None
    last_key = None

    for ts, ev, key in events:
        if ev == "KeyDown":
            if last_keydown_ts is not None:
                delay = ts - last_keydown_ts
                inter_key_delays.append(delay)

                # Assign delay to previous key
                per_key_inter[last_key].append(delay)

            last_keydown_ts = ts
            last_key = key

    # ----------------------------
    # Step 4: Typing rate
    # ----------------------------
    keydown_count = sum(1 for _, ev, _ in events if ev == "KeyDown")
    total_time = events[-1][0] - events[0][0]
    typing_rate = keydown_count / total_time if total_time > 0 else 0

    # ----------------------------
    # Step 5: Backspace count
    # ----------------------------
    backspace_count = sum(
        1 for _, ev, key in events
        if ev == "KeyDown" and "back" in key.lower()
    )

    hold_arr = np.array(hold_times)
    inter_arr = np.array(inter_key_delays)

    if len(hold_arr) == 0 or len(inter_arr) == 0:
        return None

    # ----------------------------
    # Step 6: Global features
    # ----------------------------
    features = {
        "hold_mean": hold_arr.mean(),
        "hold_std": hold_arr.std(),
        "hold_median": np.median(hold_arr),

        "inter_mean": inter_arr.mean(),
        "inter_std": inter_arr.std(),
        "inter_median": np.median(inter_arr),

        "typing_rate": typing_rate,
        "backspace_count": backspace_count,
    }

    # ----------------------------
    # Step 7: Per-key features
    # ----------------------------
    selected_keys = ["E", "T", "A", "I", "N", "S", "R", "H"]

    for k in selected_keys:
        # Hold features
        if k in per_key_holds and len(per_key_holds[k]) > 0:
            features[f"hold_mean_{k}"] = np.mean(per_key_holds[k])
            features[f"hold_std_{k}"] = np.std(per_key_holds[k])
        else:
            features[f"hold_mean_{k}"] = 0
            features[f"hold_std_{k}"] = 0

        # Inter-key features
        if k in per_key_inter and len(per_key_inter[k]) > 0:
            features[f"inter_mean_{k}"] = np.mean(per_key_inter[k])
            features[f"inter_std_{k}"] = np.std(per_key_inter[k])
        else:
            features[f"inter_mean_{k}"] = 0
            features[f"inter_std_{k}"] = 0

    return features


# ============================
# MAIN: process all files
# ============================

rows = []

for root, dirs, files in os.walk(DATA_ROOT):
    for fname in files:
        if not fname.endswith(".txt"):
            continue

        file_path = os.path.join(root, fname)

        feats = extract_features(file_path)
        if feats is None:
            continue

        rel = os.path.relpath(file_path, DATA_ROOT)
        parts = rel.split(os.sep)

        session = parts[0] if len(parts) > 1 else "unknown"
        condition = parts[1] if len(parts) > 2 else "unknown"
        file_id = os.path.splitext(parts[-1])[0]

        try:
            user_id = int(file_id[:3])
        except:
            user_id = file_id

        feats["user"] = user_id
        feats["session"] = session
        feats["condition"] = condition
        feats["file"] = file_id

        rows.append(feats)

df = pd.DataFrame(rows)

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)

print("Feature dataset saved to:", OUTPUT_CSV)
print("Shape:", df.shape)
