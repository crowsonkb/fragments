"""Embeds a NumPy array containing image data using the data URI scheme."""

import base64
import io
import numpy as np
import PIL.Image

__all__ = ['EmbedImage']


class EmbedImage:
    """Embeds a NumPy array containing image data into an HTML document,
    such as Jupyter Notebook, using the data URI scheme."""

    NEAREST_STYLE = """image-rendering: -moz-crisp-edges;
                       image-rendering:   -o-crisp-edges;
                       image-rendering: -webkit-optimize-contrast;
                       image-rendering: pixelated;
                       -ms-interpolation-mode: nearest-neighbor;"""

    def __init__(self, arr, scale=1, fmt='png', nearest=False):
        """Constructs a new EmbeddedImage object.

        Args:
            arr (np.array): The input image data.
            scale (float): The amount that the browser should scale the
                rendered image, relative to its original dimension. 1/2 or
                smaller produces rendering suitable for a HiDPI display.
            format (str): The image format to embed into the HTML document.
                Options include 'png', 'jpeg', and others supported by PIL.
            nearest (bool): If True, instructs the browser to scale the image
                using nearest neighbor (pixelated) interpolation.
        """
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        img = PIL.Image.fromarray(np.uint8(np.round(np.clip(arr, 0, 255))))
        with io.BytesIO() as buf:
            img.save(buf, format=fmt)
            data = base64.b64encode(buf.getvalue()).decode()
        style = ''
        if nearest:
            style = self.NEAREST_STYLE
        template = """<img width="%d" height="%d" style="%s"
                      src="data:image/png;base64,%s">"""
        self.html = template % (round(img.width*scale),
                                round(img.height*scale),
                                style, data)

    def _repr_html_(self):
        """Called by the Jupyter Notebook to render this image object.

        Returns:
            The HTML representation, containing an embedded image.
        """
        return self.html
