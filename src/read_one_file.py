# Step-by-step keystroke feature extraction for ONE file
# This script:
# 1. Reads one keystroke log
# 2. Parses events into (timestamp, event, key)
# 3. Computes key hold times
# 4. Computes inter-key delays
# 5. Aggregates them into statistics

import os
import numpy as np
from collections import defaultdict

# Path to one session folder
DATA_PATH = "../data/UB_keystroke_dataset/s0/baseline"

# Pick one keystroke file
files = [f for f in os.listdir(DATA_PATH) if f.endswith(".txt")]
files.sort()

sample_file = files[0]
file_path = os.path.join(DATA_PATH, sample_file)

print("Reading file:", file_path)
print("-" * 50)

# --------------------------------------------------
# Step 1: Parse keystroke lines into structured data
# --------------------------------------------------

events = []

with open(file_path, "r", encoding="latin1") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        parts = line.split()

        # Expect: <key> <KeyDown/KeyUp> <timestamp>
        if len(parts) >= 3 and parts[-1].isdigit():
            key = " ".join(parts[:-2])
            event_type = parts[-2]
            timestamp = int(parts[-1])
            events.append((timestamp, event_type, key))

# Sort events by timestamp
events.sort(key=lambda x: x[0])

print("First 10 parsed events:")
for e in events[:10]:
    print(e)

# --------------------------------------------------
# Step 2: Compute key hold times (KeyDown -> KeyUp)
# --------------------------------------------------

keydown_times = defaultdict(list)
hold_times = []

for ts, ev, key in events:
    if ev == "KeyDown":
        keydown_times[key].append(ts)
    elif ev == "KeyUp":
        if keydown_times[key]:
            down_ts = keydown_times[key].pop(0)
            hold_time = ts - down_ts
            hold_times.append((key, hold_time))

print("\nFirst 10 key hold times (key, duration):")
for ht in hold_times[:10]:
    print(ht)

# --------------------------------------------------
# Step 3: Compute inter-key delays (KeyDown -> KeyDown)
# --------------------------------------------------

inter_key_delays = []
last_keydown_ts = None

for ts, ev, key in events:
    if ev == "KeyDown":
        if last_keydown_ts is not None:
            delay = ts - last_keydown_ts
            inter_key_delays.append(delay)
        last_keydown_ts = ts

print("\nFirst 10 inter-key delays:")
for d in inter_key_delays[:10]:
    print(d)

# --------------------------------------------------
# Step 4: Aggregate into statistical features
# --------------------------------------------------

hold_array = np.array([ht for _, ht in hold_times])
inter_array = np.array(inter_key_delays)

print("\nAggregated features:")

print("\nHold time:")
print("  mean   =", hold_array.mean())
print("  std    =", hold_array.std())
print("  median =", np.median(hold_array))

print("\nInter-key delay:")
print("  mean   =", inter_array.mean())
print("  std    =", inter_array.std())
print("  median =", np.median(inter_array))
