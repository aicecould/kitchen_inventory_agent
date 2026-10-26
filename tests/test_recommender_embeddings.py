import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from recommender.embeddings import (
    EMBEDDING_MAX_SEQ_LENGTH,
    EMBEDDING_MODEL,
    EMBEDDING_MODEL_REVISION,
    LocalSentenceEncoder,
    _initialize_wsl_xpu,
    load_model_manifest,
)
from recommender.models import RecommenderArtifactsError


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def make_fake_model(
    tmp_path: Path,
    *,
    model_id: str = EMBEDDING_MODEL,
    revision: str = EMBEDDING_MODEL_REVISION,
    file_map: dict[str, str] | None = None,
) -> Path:
    model_path = tmp_path / "model"
    pooling = model_path / "1_Pooling"
    pooling.mkdir(parents=True)
    files = file_map or {
        "config.json": '{"model_type":"bert"}',
        "model.safetensors": "fake-weights",
        "tokenizer.json": '{"tokenizer":"fake"}',
        "modules.json": '["fake"]',
        "1_Pooling/config.json": '{"pooling":"mean"}',
    }
    manifest_files: dict[str, str] = {}
    for relative, content in files.items():
        target = (model_path / relative).resolve()
        target.write_text(content, encoding="utf-8")
        manifest_files[relative] = _digest(target.read_bytes())
    (model_path / "model-manifest.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "revision": revision,
                "files": manifest_files,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return model_path


def test_encoder_constants_use_bge_small() -> None:
    assert EMBEDDING_MODEL == "BAAI/bge-small-en-v1.5"
    assert EMBEDDING_MODEL_REVISION == (
        "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    )
    assert EMBEDDING_MAX_SEQ_LENGTH == 512
    assert LocalSentenceEncoder.model_id == EMBEDDING_MODEL


def test_encode_query_keeps_batch_dimension_for_faiss() -> None:
    encoder = object.__new__(LocalSentenceEncoder)
    encoder._encode = lambda texts, batch_size: np.zeros(
        (1, 3), dtype=np.float32
    )

    query = encoder.encode_query("chicken rice")

    assert query.shape == (1, 3)


def test_load_model_manifest_accepts_pinned_model(tmp_path: Path) -> None:
    model_path = make_fake_model(tmp_path)

    manifest = load_model_manifest(model_path)

    assert manifest.model_id == EMBEDDING_MODEL
    assert manifest.revision == EMBEDDING_MODEL_REVISION
    assert set(manifest.files) == {
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "modules.json",
        "1_Pooling/config.json",
    }


def test_load_model_manifest_rejects_missing_key_file(tmp_path: Path) -> None:
    model_path = make_fake_model(tmp_path)
    (model_path / "tokenizer.json").unlink()

    with pytest.raises(RecommenderArtifactsError):
        load_model_manifest(model_path)


def test_load_model_manifest_rejects_sha256_mismatch(tmp_path: Path) -> None:
    model_path = make_fake_model(tmp_path)
    (model_path / "config.json").write_text("tampered", encoding="utf-8")

    with pytest.raises(RecommenderArtifactsError):
        load_model_manifest(model_path)


@pytest.mark.parametrize(
    ("model_id", "revision"),
    [
        ("BAAI/wrong-model", EMBEDDING_MODEL_REVISION),
        (EMBEDDING_MODEL, "wrong-revision"),
    ],
)
def test_load_model_manifest_rejects_wrong_identity(
    tmp_path: Path,
    model_id: str,
    revision: str,
) -> None:
    model_path = make_fake_model(tmp_path, model_id=model_id, revision=revision)

    with pytest.raises(RecommenderArtifactsError):
        load_model_manifest(model_path)


@pytest.mark.parametrize(
    "relative",
    [
        "../tokenizer.json",
        "/tmp/tokenizer.json",
    ],
)
def test_load_model_manifest_rejects_unsafe_file_paths(
    tmp_path: Path,
    relative: str,
) -> None:
    model_path = make_fake_model(tmp_path)
    manifest_path = model_path / "model-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = {
        **manifest["files"],
        relative: "0" * 64,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RecommenderArtifactsError):
        load_model_manifest(model_path)


class RecordingSentenceTransformer:
    def __init__(self) -> None:
        self.max_seq_length = 512
        self.encoded_texts: list[list[str]] = []

    def encode(self, sentences: list[str], **kwargs: object) -> np.ndarray:
        self.encoded_texts.append(list(sentences))
        return np.zeros((len(sentences), 3), dtype=np.float32)


def test_encode_query_uses_bge_instruction_without_changing_documents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = make_fake_model(tmp_path)
    transformer = RecordingSentenceTransformer()
    monkeypatch.setattr(
        "recommender.embeddings._initialize_wsl_xpu", lambda: None
    )
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=lambda *args, **kwargs: transformer),
    )

    encoder = LocalSentenceEncoder(model_path, local_files_only=True)
    encoder.encode_query("chicken rice")
    encoder.encode_documents(["chicken rice"])

    assert transformer.encoded_texts == [
        ["Represent this sentence for searching relevant passages: chicken rice"],
        ["chicken rice"],
    ]


@pytest.mark.parametrize(
    "release,xpu_version,expected",
    [
        ("6.18-microsoft-standard-WSL2", "20260100", ["init"]),
        ("6.18-microsoft-standard-WSL2", None, []),
        ("6.18-generic", "20260100", []),
    ],
)
def test_wsl_initializes_xpu_without_sysman_probe(
    monkeypatch: pytest.MonkeyPatch,
    release: str,
    xpu_version: str | None,
    expected: list[str],
) -> None:
    calls: list[str] = []
    monkeypatch.setattr("recommender.embeddings.platform.release", lambda: release)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            version=SimpleNamespace(xpu=xpu_version),
            xpu=SimpleNamespace(init=lambda: calls.append("init")),
        ),
    )
    _initialize_wsl_xpu()
    assert calls == expected
