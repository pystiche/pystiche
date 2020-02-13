from .transforms import (
    ComposedTransform,
    Normalize,
    Denormalize,
    FloatToUint8Range,
    Uint8ToFloatRange,
    ReverseChannelOrder,
)


TORCH_MEAN = (0.485, 0.456, 0.406)
TORCH_STD = (0.229, 0.224, 0.225)

CAFFE_MEAN = (0.485, 0.458, 0.408)
CAFFE_STD = (1.0, 1.0, 1.0)

__all__ = [
    "TorchPreprocessing",
    "TorchPostprocessing",
    "torch_processing",
    "CaffePreprocessing",
    "CaffePostprocessing",
    "caffe_processing",
]


def _processing(preprocessor=None, postprocessor=None):
    def outer_wrapper(fn):
        def inner_wrapper(input, *args, **kwargs):
            if preprocessor is not None:
                input = preprocessor(input)

            output = fn(input, *args, **kwargs)

            if postprocessor is not None:
                output = postprocessor(output)

            return output

        return inner_wrapper

    return outer_wrapper


class TorchPreprocessing(ComposedTransform):
    def __init__(self) -> None:
        transforms = (Normalize(TORCH_MEAN, TORCH_STD),)
        super().__init__(*transforms)


class TorchPostprocessing(ComposedTransform):
    def __init__(self) -> None:
        transforms = (Denormalize(TORCH_MEAN, TORCH_STD),)
        super().__init__(*transforms)


def torch_processing(fn):
    preprocessor = TorchPreprocessing()
    postprocessor = TorchPostprocessing()

    processing = _processing(preprocessor, postprocessor)
    return processing(fn)


class CaffePreprocessing(ComposedTransform):
    def __init__(self) -> None:
        transforms = (
            Normalize(CAFFE_MEAN, CAFFE_STD),
            FloatToUint8Range(),
            ReverseChannelOrder(),
        )
        super().__init__(*transforms)


class CaffePostprocessing(ComposedTransform):
    def __init__(self) -> None:
        transforms = (
            ReverseChannelOrder(),
            Uint8ToFloatRange(),
            Denormalize(CAFFE_MEAN, CAFFE_STD),
        )
        super().__init__(*transforms)


def caffe_processing(fn):
    preprocessor = CaffePreprocessing()
    postprocessor = CaffePostprocessing()

    processing = _processing(preprocessor, postprocessor)
    return processing(fn)
