#!/usr/bin/env python3
"""Sample ChartNet human_verified with Hugging Face Datasets and save locally.

Train: 4000 random rows. Test: 1000 random rows.
Each Hub endpoint is tried in its own process, because huggingface_hub reads
HF_ENDPOINT once at import. Streaming plus reservoir sampling keeps only the
sample in memory.
"""

import os
import random
import subprocess
import sys
from pathlib import Path

REPO_ID = "ibm-granite/ChartNet"
CONFIG = "human_verified"
SEED = 42
TRAIN_N = 4000
TEST_N = 1000
OUT_DIR = Path(__file__).resolve().parent / "data" / "chartnet_human_verified_sample"
CHILD_FLAG = "CHARTNET_SAMPLE_CHILD"


def reservoir_sample(stream, k, seed, split_name):
    rng = random.Random(seed)
    sample = []
    seen = 0
    for seen, row in enumerate(stream, start=1):
        if len(sample) < k:
            sample.append(row)
        else:
            j = rng.randrange(seen)
            if j < k:
                sample[j] = row
        if seen % 200 == 0:
            print(f"[{split_name}] scanned {seen}, kept {len(sample)}", flush=True)
    if seen == 0:
        raise RuntimeError(f"{split_name} split is empty")
    if len(sample) < k:
        print(
            f"[{split_name}] only {len(sample)} rows available, requested {k}",
            flush=True,
        )
    print(f"[{split_name}] done, scanned {seen}, sample {len(sample)}", flush=True)
    return sample


def save_sample(rows, features, out_path):
    from datasets import Dataset

    out_path.mkdir(parents=True, exist_ok=True)

    def gen():
        yield from rows

    ds = Dataset.from_generator(gen, features=features)
    ds.save_to_disk(str(out_path))
    print(f"saved {len(ds)} rows -> {out_path}", flush=True)


def download():
    from datasets import load_dataset

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    print(f"output: {OUT_DIR}", flush=True)
    print(f"endpoint: {endpoint}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, k in (("test", TEST_N), ("train", TRAIN_N)):
        print(f"loading {split}", flush=True)
        stream = load_dataset(REPO_ID, CONFIG, split=split, streaming=True)
        features = stream.features
        print(f"[{split}] columns: {list(features)}", flush=True)
        rows = reservoir_sample(stream, k, SEED, split)
        save_sample(rows, features, OUT_DIR / split)
    print("finished", flush=True)


def endpoints_to_try():
    chosen = os.environ.get("HF_ENDPOINT")
    if chosen:
        return [chosen]
    return ["https://hf-mirror.com", "https://huggingface.co"]


def main():
    if os.environ.get(CHILD_FLAG) == "1":
        download()
        return

    errors = []
    for endpoint in endpoints_to_try():
        env = os.environ.copy()
        env["HF_ENDPOINT"] = endpoint
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        env[CHILD_FLAG] = "1"
        print(f"trying {endpoint}", flush=True)
        completed = subprocess.run([sys.executable, __file__], env=env)
        if completed.returncode == 0:
            return
        errors.append(f"{endpoint} exited {completed.returncode}")
    raise RuntimeError("all endpoints failed: " + "; ".join(errors))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
