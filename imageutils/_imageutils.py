import base64
import io
import numpy as np
import PIL.Image

RESAMPLERS = {
    'nearest': PIL.Image.NEAREST,
    'bilinear': PIL.Image.BILINEAR,
    'bicubic': PIL.Image.BICUBIC,
    'lanczos': PIL.Image.LANCZOS
}


def _resample_plane(plane, h, w, method):
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
            'bicubic':  cubic spline resampling (best for resampling up)
            'lanczos':  Lanczos filtering (best for resampling down)

    Returns:
        np.array: The output image.
    """
    resampler = RESAMPLERS[method]
    if arr.ndim == 2:
        return _resample_plane(arr, h, w, resampler)
    if arr.ndim == 3:
        planes = []
        for i in range(arr.shape[-1]):
            planes.append(arr[..., i])
    else:
        raise ValueError('Only HxW and HxWxC arrays are supported')
    new_planes = []
    for plane in planes:
        new_planes.append(_resample_plane(plane, h, w, resampler))
    return np.stack(new_planes, axis=2)


class EmbeddedImage:
    """Embeds a NumPy array containing image data into an HTML document,
    such as Jupyter Notebook, using the data URI scheme."""

    def __init__(self, arr, scale=1, format='png', nearest=False):
        """Args:
            arr (np.array): The input image data.
            scale (float): The amount that the browser should scale the rendered image,
                relative to its original dimension. 1/2 should produce image rendering
                suitable for a HiDPI display.
            format (str): The image format to embed into the HTML document. Options include
                'png', 'jpeg', and others supported by PIL.
            nearest (bool): If True, instructs the browser to scale the image using
                nearest neighbor (pixelated) interpolation.
        """
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        img = PIL.Image.fromarray(np.uint8(np.round(np.clip(arr, 0, 255))))
        with io.BytesIO() as buf:
            img.save(buf, format=format)
            data = base64.b64encode(buf.getvalue()).decode()
        style = ''
        if nearest:
            # TODO: make this work in more than just Webkit
            style = 'image-rendering: -webkit-optimize-contrast;'
        template = '<img width="%d" height="%d" style="%s" src="data:image/png;base64,%s">'
        self.html = template % (round(img.width*scale), round(img.height*scale), style, data)

    def _repr_html_(self):
        """Called by the Jupyter Notebook to render this EmbeddedImage object."""
        return self.html
