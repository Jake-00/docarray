from typing import Dict, Generic, Type

from docarray.typing.tensor.abstract_tensor import AbstractTensor, T
from docarray.typing.tensor.audio.abstract_audio_tensor import AbstractAudioTensor
from docarray.typing.tensor.audio.audio_ndarray import AudioNdArray
from docarray.typing.tensor.tensor import NdArray
from docarray.utils.misc import is_tf_available, is_torch_available

_AUDIO_TENSOR: Dict[Type[AbstractTensor], Type[AbstractAudioTensor]] = {
    NdArray: AudioNdArray,
}

if is_torch_available():
    from docarray.typing.tensor.audio.audio_torch_tensor import AudioTorchTensor
    from docarray.typing.tensor.tensor import TorchTensor

    _AUDIO_TENSOR[TorchTensor] = AudioTorchTensor

if is_tf_available():
    from docarray.typing.tensor.audio.audio_tensorflow_tensor import (
        AudioTensorFlowTensor as AudioTFTensor,
    )
    from docarray.typing.tensor.tensor import TensorFlowTensor

    _AUDIO_TENSOR[TensorFlowTensor] = AudioTFTensor


class AudioTensor(AudioNdArray, Generic[T]):
    @classmethod
    def __class_getitem__(cls, item: Type[AbstractTensor]) -> Type[AbstractAudioTensor]:
        try:
            return _AUDIO_TENSOR[item]
        except Exception:
            return AudioTorchTensor
