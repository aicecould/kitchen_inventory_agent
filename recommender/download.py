"""Download the pinned embedding model and recipe CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

from recommender.embeddings import load_model_manifest
from recommender.model_config import EMBEDDING_MODEL, EMBEDDING_MODEL_REVISION
from recommender.build import SOURCE_COMMIT


ROOT = Path(__file__).resolve().parent
MODEL_FILES = (
    "config.json", "model.safetensors", "tokenizer.json", "modules.json",
    "1_Pooling/config.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.txt", "sentence_bert_config.json", "config_sentence_transformers.json",
)


def download_file(url: str, target: Path) -> None:
    if target.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".download")
    print(f"Downloading {target.name}", flush=True)
    urlretrieve(url, temporary)
    temporary.replace(target)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    model_dir = ROOT / "model"
    for name in MODEL_FILES:
        download_file(
            f"https://huggingface.co/{EMBEDDING_MODEL}/resolve/"
            f"{EMBEDDING_MODEL_REVISION}/{name}",
            model_dir / name,
        )
    load_model_manifest(model_dir)
    download_file(
        "https://raw.githubusercontent.com/josephrmartinez/recipe-dataset/"
        f"{SOURCE_COMMIT}/13k-recipes.csv",
        ROOT / "dataset" / "13k-recipes.csv",
    )
    print("Model verified; recipe CSV ready.")


if __name__ == "__main__":
    main()
