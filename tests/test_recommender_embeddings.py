import numpy as np

from recommender.embeddings import (
    EMBEDDING_MAX_SEQ_LENGTH,
    EMBEDDING_MODEL,
    EMBEDDING_MODEL_REVISION,
    LocalSentenceEncoder,
)


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
