import numpy as np

from docarray import BaseDocument
from docarray.typing import EmbeddingTensor


def test_set_embedding():
    class MyDocument(BaseDocument):
        embedding: EmbeddingTensor

    d = MyDocument(embedding=np.zeros((3, 224, 224)))

    assert isinstance(d.embedding, np.ndarray)
    assert (d.embedding == np.zeros((3, 224, 224))).all()
