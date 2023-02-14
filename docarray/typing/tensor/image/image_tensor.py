from typing import Dict, Generic, Type

from docarray.typing.tensor.abstract_tensor import AbstractTensor, T
from docarray.typing.tensor.image.abstract_image_tensor import AbstractImageTensor
from docarray.typing.tensor.image.image_ndarray import ImageNdArray
from docarray.typing.tensor.tensor import NdArray
from docarray.utils.misc import is_tf_available, is_torch_available

_IMAGE_TENSOR: Dict[Type[AbstractTensor], Type[AbstractImageTensor]] = {
    NdArray: ImageNdArray,
}

if is_torch_available():
    from docarray.typing.tensor.image.image_torch_tensor import ImageTorchTensor
    from docarray.typing.tensor.tensor import TorchTensor

    _IMAGE_TENSOR[TorchTensor] = ImageTorchTensor


tf_available = is_tf_available()
if tf_available:
    from docarray.typing.tensor.image.image_tensorflow_tensor import (
        ImageTensorFlowTensor as ImageTFTensor,
    )
    from docarray.typing.tensor.tensor import TensorFlowTensor

    _IMAGE_TENSOR[TensorFlowTensor] = ImageTFTensor


class ImageTensor(Generic[T]):
    def __class_getitem__(
        self, item: Type[AbstractTensor]
    ) -> Type[AbstractImageTensor]:
        return _IMAGE_TENSOR[item]
