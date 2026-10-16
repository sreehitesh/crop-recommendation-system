"""Download the Crop Recommendation Dataset from public mirrors.

Primary source (Kaggle): https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
Kaggle requires an account/API token, so this script falls back to well-known
public mirrors of the same file on GitHub and verifies that at least two
mirrors produce identical content before saving the dataset locally.
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TARGET = DATA_DIR / "Crop_recommendation.csv"

MIRRORS = [
    "https://raw.githubusercontent.com/Gladiator07/Harvestify/master/Data-processed/crop_recommendation.csv",
    "https://raw.githubusercontent.com/Gyanalyst/Optimizing-Agricultural-Production/main/Crop_recommendation.csv",
    "https://raw.githubusercontent.com/Bear2codes/Upskillcampus/main/Crop_recommendation.csv",
    "https://raw.githubusercontent.com/Thisurika/Crop-Recommendation-AI/main/Crop_recommendation.csv",
]


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def main() -> int:
    if TARGET.exists():
        print(f"Dataset already present: {TARGET}")
        return 0

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    fetched = []  # list of (url, sha256, bytes)
    for url in MIRRORS:
        try:
            raw = fetch(url)
        except Exception as exc:  # noqa: BLE001 - report and try next mirror
            print(f"[skip] {url}\n       {exc}")
            continue
        if not raw.startswith(b"N,P,K,temperature"):
            print(f"[skip] {url} (unexpected header: {raw[:60]!r})")
            continue
        digest = hashlib.sha256(raw).hexdigest()
        print(f"[ok]   {url} -> {len(raw)} bytes, sha256={digest[:16]}...")
        fetched.append((url, digest, raw))
        if len(fetched) == 2:
            break

    if not fetched:
        print("ERROR: no mirror could be downloaded.")
        return 1

    digests = {d for _, d, _ in fetched}
    if len(fetched) > 1 and len(digests) > 1:
        print("WARNING: mirrors differ; saving the first verified mirror.")

    url, digest, raw = fetched[0]
    TARGET.write_bytes(raw)
    print(f"Saved {TARGET} from {url}")
    print(f"sha256: {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
