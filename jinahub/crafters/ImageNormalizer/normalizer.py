__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from pydoc import locate
from typing import Tuple, Union, Iterable
import numpy as np
import PIL.Image as Image

from jina import DocumentArray, Executor, requests
from jina_commons import get_logger


class ImageNormalizer(Executor):
    def __init__(
        self,
        target_size: Union[Iterable[int], int] = 224,
        img_mean: Tuple[float] = (0, 0, 0),
        img_std: Tuple[float] = (1, 1, 1),
        resize_dim: Union[Iterable[int], int] = 256,
        channel_axis: int = -1,
        target_channel_axis: int = -1,
        target_dtype: Union[str,np.dtype] = np.float32,
        *args,
        **kwargs,
    ):
        """Set Constructor."""
        super().__init__(*args, **kwargs)
        self.target_size = target_size
        self.resize_dim = resize_dim
        self.img_mean = np.array(img_mean).reshape((1, 1, 3))
        self.img_std = np.array(img_std).reshape((1, 1, 3))
        self.channel_axis = channel_axis
        self.target_channel_axis = target_channel_axis
        self.logger = get_logger(self)

        # when passed from yaml it is string
        if isinstance(target_dtype, str):
            actual_type = locate(target_dtype)
            if actual_type:
                self.target_dtype = actual_type
            else:
                self.logger.error(f'Could not resolve type "{target_dtype}". '
                                  f'Make sure you use "numpy.float32"-like syntax')

        else:
            self.target_dtype = target_dtype


    @requests
    def craft(self, docs: DocumentArray, **kwargs) -> DocumentArray:
        filtered_docs = DocumentArray(
            list(filter(lambda d: 'image/' in d.mime_type, docs))
        )

        for doc in filtered_docs:
            self._convert_image_to_blob(doc)
            raw_img = self._load_image(doc.blob)
            _img = self._normalize(raw_img)
            # move the channel_axis to target_channel_axis to better fit
            # different models
            img = self._move_channel_axis(_img, -1, self.target_channel_axis)
            doc.blob = img.astype(self.target_dtype)
        return filtered_docs

    def _convert_image_to_blob(self, doc):
        if doc.uri:
            doc.convert_image_uri_to_blob()
        elif doc.buffer:
            doc.convert_image_buffer_to_blob()

    def _normalize(self, img):
        img = self._resize_short(img)
        img, _, _ = self._crop_image(img, how='center')
        img = np.array(img).astype('float32') / 255
        img -= self.img_mean
        img /= self.img_std
        return img

    def _load_image(self, blob: 'np.ndarray'):
        """
        Load an image array and return a `PIL.Image` object.
        """
        img = self._move_channel_axis(blob, self.channel_axis)
        return Image.fromarray(img.astype('uint8'))

    @staticmethod
    def _move_channel_axis(
        img: 'np.ndarray', channel_axis_to_move: int, target_channel_axis: int = -1
    ) -> 'np.ndarray':
        """
        Ensure the color channel axis is the default axis.
        """
        if channel_axis_to_move == target_channel_axis:
            return img
        return np.moveaxis(img, channel_axis_to_move, target_channel_axis)

    def _crop_image(self, img, top: int = None, left: int = None, how: str = 'precise'):
        """
        Crop the input :py:mod:`PIL` image.
        :param img: :py:mod:`PIL.Image`, the image to be resized
        :param target_size: desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the output will have the same height and width as the `target_size`.
        :param top: the vertical coordinate of the top left corner of the crop box.
        :param left: the horizontal coordinate of the top left corner of the crop box.
        :param how: the way of cropping. Valid values include `center`, `random`, and, `precise`. Default is `precise`.
            - `center`: crop the center part of the image
            - `random`: crop a random part of the image
            - `precise`: crop the part of the image specified by the crop box with the given ``top`` and ``left``.
            .. warning:: When `precise` is used, ``top`` and ``left`` must be fed valid value.
        """
        assert isinstance(img, Image.Image), 'img must be a PIL.Image'
        img_w, img_h = img.size
        if isinstance(self.target_size, int):
            target_h = target_w = self.target_size
        elif isinstance(self.target_size, Tuple) and len(self.target_size) == 2:
            target_h, target_w = self.target_size
        else:
            raise ValueError(
                f'target_size should be an integer or a tuple of '
                f'two integers: {self.target_size}'
            )
        w_beg = left
        h_beg = top
        if how == 'center':
            w_beg = int((img_w - target_w) / 2)
            h_beg = int((img_h - target_h) / 2)
        elif how == 'random':
            w_beg = np.random.randint(0, img_w - target_w + 1)
            h_beg = np.random.randint(0, img_h - target_h + 1)
        elif how == 'precise':
            assert w_beg is not None and h_beg is not None
            assert (
                0 <= w_beg <= (img_w - target_w)
            ), f'left must be within [0, {img_w - target_w}]: {w_beg}'
            assert (
                0 <= h_beg <= (img_h - target_h)
            ), f'top must be within [0, {img_h - target_h}]: {h_beg}'
        else:
            raise ValueError(f'unknown input how: {how}')
        if not isinstance(w_beg, int):
            raise ValueError(f'left must be int number between 0 and {img_w}: {left}')
        if not isinstance(h_beg, int):
            raise ValueError(f'top must be int number between 0 and {img_h}: {top}')
        w_end = w_beg + target_w
        h_end = h_beg + target_h
        img = img.crop((w_beg, h_beg, w_end, h_end))
        return img, h_beg, w_beg

    def _resize_short(self, img, how: str = 'LANCZOS'):
        """
        Resize the input :py:mod:`PIL` image.
        :param img: :py:mod:`PIL.Image`, the image to be resized
        :param target_size: desired output size. If size is a sequence like (h, w), the output size will be matched to
            this. If size is an int, the smaller edge of the image will be matched to this number maintain the aspect
            ratio.
        :param how: the interpolation method. Valid values include `NEAREST`, `BILINEAR`, `BICUBIC`, and `LANCZOS`.
            Default is `LANCZOS`. Please refer to `PIL.Image` for detaisl.
        """
        assert isinstance(img, Image.Image), 'img must be a PIL.Image'
        if isinstance(self.resize_dim, int):
            percent = float(self.resize_dim) / min(img.size[0], img.size[1])
            target_w = int(round(img.size[0] * percent))
            target_h = int(round(img.size[1] * percent))
        elif isinstance(self.resize_dim, Tuple) and len(self.resize_dim) == 2:
            target_w, target_h = self.resize_dim
        else:
            raise ValueError(
                f'target_size should be an integer or a tuple of two '
                f'integers: {self.resize_dim}'
            )
        img = img.resize((target_w, target_h), getattr(Image, how))
        return img
