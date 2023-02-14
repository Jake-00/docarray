from typing import Dict, Generic, Type

from docarray.typing.tensor.abstract_tensor import AbstractTensor, T
from docarray.typing.tensor.tensor import NdArray
from docarray.typing.tensor.video.video_ndarray import VideoNdArray
from docarray.typing.tensor.video.video_tensor_mixin import VideoTensorMixin
from docarray.utils.misc import is_tf_available, is_torch_available

_VIDEO_TENSOR: Dict[Type[AbstractTensor], Type[VideoTensorMixin]] = {
    NdArray: VideoNdArray,
}

if is_torch_available():
    from docarray.typing.tensor import TorchTensor
    from docarray.typing.tensor.video.video_torch_tensor import VideoTorchTensor

    _VIDEO_TENSOR[TorchTensor] = VideoTorchTensor


if is_tf_available():
    from docarray.typing.tensor import TensorFlowTensor
    from docarray.typing.tensor.video.video_tensorflow_tensor import (
        VideoTensorFlowTensor as VideoTFTensor,
    )

    _VIDEO_TENSOR[TensorFlowTensor] = VideoTFTensor


class VideoTensor(Generic[T]):
    def __class_getitem__(self, item: Type[AbstractTensor]) -> Type[VideoTensorMixin]:
        return _VIDEO_TENSOR[item]
