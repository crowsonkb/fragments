"""Resamples a NumPy array containing image data using PIL."""

import numpy as np
import PIL.Image

__all__ = ['resample']

RESAMPLERS = {
    'nearest': PIL.Image.NEAREST,
    'bilinear': PIL.Image.BILINEAR,
    'bicubic': PIL.Image.BICUBIC,
    'lanczos': PIL.Image.LANCZOS
}


def resample_plane(plane, h, w, method):
    """Resamples a single-channel HxW plane of image data."""
    return np.array(PIL.Image.fromarray(plane).resize((w, h), method))


def resample(arr, h, w, method='bicubic'):
    """Resamples a NumPy array containing image data using PIL. Allowed input
    dtypes are: uint8, int32, float32, and float64 (converted to float32).

    Args:
        arr (np.array): The input image data, in HxW or HxWxC (packed) formats.
        h (int): The height of the output image, in pixels.
        w (int): The width of the output image, in pixels.
        method (str): The resampling method to use.
            'nearest':  nearest neighbor (pixelated)
            'bilinear': linear interpolation
            'bicubic':  cubic spline interpolation (default)
            'lanczos':  Lanczos interpolation (best for resampling down)

    Returns:
        np.array: The output image.
    """
    resampler = RESAMPLERS[method]
    if arr.ndim == 2:
        return resample_plane(arr, h, w, resampler)
    if arr.ndim == 3:
        planes = []
        for i in range(arr.shape[-1]):
            planes.append(arr[..., i])
    else:
        raise ValueError('Only HxW and HxWxC arrays are supported')
    new_planes = []
    for plane in planes:
        new_planes.append(resample_plane(plane, h, w, resampler))
    return np.stack(new_planes, axis=2)
