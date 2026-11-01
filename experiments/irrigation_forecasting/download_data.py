"""Download only the soil-moisture Parquet files needed for this experiment."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path


REVISION = "1580f0627902a29bf563f31f7174c3f59ee1713b"
FILES = {
    "field_sensor_data_consortium0.parquet": (
        "7dbdd398cdeda6d62bd39446d91d62398be44c5b3ec4619b28c1e0e305ed9e2a"
    ),
    "field_sensor_data_consortium2.parquet": (
        "ba4a76219b6d746769e6120920f343f36e28e8778f2b88f85d401929f5ef3484"
    ),
}
BASE_URL = (
    "https://huggingface.co/datasets/it4lia/soil_moisture_dataset/resolve/"
    f"{REVISION}"
)
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_data(data_dir: Path) -> list[Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for filename, expected_hash in FILES.items():
        target = data_dir / filename
        if target.exists():
            actual_hash = sha256_file(target)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Existing file failed SHA-256 verification: {target}"
                )
        else:
            request = urllib.request.Request(
                f"{BASE_URL}/{filename}",
                headers={"User-Agent": "Schisto-AI-Training-experiment/1.0"},
            )
            temporary = target.with_suffix(target.suffix + ".part")
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    with temporary.open("wb") as handle:
                        while chunk := response.read(1024 * 1024):
                            handle.write(chunk)
                actual_hash = sha256_file(temporary)
                if actual_hash != expected_hash:
                    raise ValueError(
                        f"Downloaded file failed SHA-256 verification: {filename}"
                    )
                temporary.replace(target)
            finally:
                if temporary.exists():
                    temporary.unlink()
        downloaded.append(target)
        print(f"Verified {target}")
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    download_data(args.data_dir)


if __name__ == "__main__":
    main()
