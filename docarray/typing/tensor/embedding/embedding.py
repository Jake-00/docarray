from typing import Dict, Generic, Type

from docarray.typing.tensor.abstract_tensor import AbstractTensor, T
from docarray.typing.tensor.embedding.embedding_mixin import EmbeddingMixin
from docarray.typing.tensor.embedding.ndarray import NdArrayEmbedding
from docarray.typing.tensor.tensor import NdArray
from docarray.utils.misc import is_tf_available, is_torch_available

_EMBEDDING_TENSOR: Dict[Type[AbstractTensor], Type[EmbeddingMixin]] = {
    NdArray: NdArrayEmbedding,
}

if is_torch_available():
    from docarray.typing.tensor.embedding.torch import TorchEmbedding
    from docarray.typing.tensor.tensor import TorchTensor

    _EMBEDDING_TENSOR[TorchTensor] = TorchEmbedding


if is_tf_available():
    from docarray.typing.tensor.embedding.tensorflow import (
        TensorFlowEmbedding as TFEmbedding,
    )
    from docarray.typing.tensor.tensor import TensorFlowTensor

    _EMBEDDING_TENSOR[TensorFlowTensor] = TFEmbedding


class EmbeddingTensor(Generic[T]):
    def __class_getitem__(self, item: Type[AbstractTensor]) -> Type[EmbeddingMixin]:
        return _EMBEDDING_TENSOR[item]
