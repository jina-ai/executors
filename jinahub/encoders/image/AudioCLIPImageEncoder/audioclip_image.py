__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Iterable, Optional

import torch
from jina import DocumentArray, Executor, requests
from jina_commons.batching import get_docs_batch_generator
from PIL import Image
from torchvision import transforms

from .audio_clip.model import AudioCLIP

# Defaults from CLIP
_IMAGE_SIZE = 224
_IMAGE_MEAN = 0.48145466, 0.4578275, 0.40821073
_IMAGE_STD = 0.26862954, 0.26130258, 0.27577711


class AudioCLIPImageEncoder(Executor):
    """
    Encode image data with the AudioCLIP model

    :param model_path: path of the pre-trained AudioCLIP model.
    :param default_traversal_paths: default traversal path (used if not specified in
        request's parameters)
    :param default_batch_size: default batch size (used if not specified in
        request's parameters)
    :param use_default_preprocessing: Whether to use the default preprocessing on
        images (blobs) before encoding them. If you disable this, you must ensure
        that the images you pass in have the correct format, see the ``encode`` method
        for details.
    :param device: device that the model is on (should be "cpu", "cuda" or "cuda:X",
        where X is the index of the GPU on the machine)
    """

    def __init__(
        self,
        model_path: str = '.cache/AudioCLIP-Full-Training.pt',
        default_traversal_paths: Iterable[str] = None,
        default_batch_size: int = 32,
        use_default_preprocessing: bool = True,
        device: str = 'cpu',
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.model = AudioCLIP(pretrained=model_path).to(device).eval()
        self.default_traversal_paths = default_traversal_paths or ['r']
        self.default_batch_size = default_batch_size
        self.use_default_preprocessing = use_default_preprocessing

        self._default_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(_IMAGE_SIZE, interpolation=Image.BICUBIC),
                transforms.CenterCrop(_IMAGE_SIZE),
                transforms.Normalize(_IMAGE_MEAN, _IMAGE_STD),
            ]
        )

    @requests
    def encode(
        self, docs: Optional[DocumentArray], parameters: dict, *args, **kwargs
    ) -> None:
        """
        Method to create embedddings for documents by encoding their image.

        :param docs: A document array with documents to create embeddings for. Only the
            documents that have the ``blob`` attribute will get embeddings. The ``blob``
            attribute should be the numpy array of the image, and should have dtype
            ``np.uint8`` (unless you set ``use_default_preprocessing=True``, then they
            can also be of a float type).

            If you set ``use_default_preprocessing=True`` when creating this encoder,
            then the image arrays should have the shape ``[H, W, C]``, and be in the
            RGB color format.

            If you set ``use_default_preprocessing=False`` when creating this encoder,
            then you need to ensure that the images you pass in are already
            pre-processed. This means that they are all the same size (for batching) -
            the CLIP model was trained on ``224 x 224`` images, and that they are of
            the shape ``[C, H, W]`` (in the RGB color format). They should also be
            normalized.
        :param parameters: A dictionary that contains parameters to control encoding.
            The accepted keys are ``traversal_paths`` and ``batch_size`` - in their
            absence their corresponding default values are used.
        """

        batch_generator = get_docs_batch_generator(
            docs,
            traversal_path=parameters.get(
                'traversal_paths', self.default_traversal_paths
            ),
            batch_size=parameters.get('batch_size', self.default_batch_size),
            needs_attr='blob',
        )

        with torch.no_grad():
            for batch in batch_generator:
                images = []
                for doc in batch:
                    if self.use_default_preprocessing:
                        if doc.blob.shape[2] != 3:
                            raise ValueError(
                                "If `use_default_preprocessing=True`, your image must"
                                " be of the format [H, W, C], in the RGB format (C=3),"
                                f" but got C={doc.blob.shape[2]} instead."
                            )
                        images.append(self._default_transforms(doc.blob.copy()))
                    else:
                        if doc.blob.shape[0] != 3:
                            raise ValueError(
                                "If `use_default_preprocessing=False`, your image must"
                                " be of the format [C, H, W], in the RGB format (C=3),"
                                f" but got C={doc.blob.shape[0]} instead."
                            )

                        images.append(torch.tensor(doc.blob, dtype=torch.float32))

                images = torch.stack(images)
                embeddings = self.model.encode_image(image=images)
                embeddings = embeddings.cpu().numpy()

                for idx, doc in enumerate(batch):
                    doc.embedding = embeddings[idx]
