__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Iterable, Optional

import torch
from jina import Executor, DocumentArray, requests
from jina_commons.batching import get_docs_batch_generator

from audio_clip.model import AudioCLIP


class AudioCLIPTextEncoder(Executor):
    """
    Encode text data with the AudioCLIP model

    :param model_path: path of the pre-trained AudioCLIP model.
    :param default_traversal_paths: default traversal path (used if not specified in
        request's parameters)
    :param default_batch_size: default batch size (used if not specified in
        request's parameters)
    :param device: device that the model is on (should be "cpu", "cuda" or "cuda:X",
        where X is the index of the GPU on the machine)
    """

    def __init__(
        self,
        model_path: str = '.cache/AudioCLIP-Full-Training.pt',
        default_traversal_paths: Iterable[str] = ['r'],
        default_batch_size: int = 32,
        device: str = 'cpu',
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.model = AudioCLIP(pretrained=model_path).to(device).eval()
        self.default_traversal_paths = default_traversal_paths
        self.default_batch_size = default_batch_size

    @requests
    def encode(
        self, docs: Optional[DocumentArray], parameters: dict, *args, **kwargs
    ) -> None:
        """
        Method to create embedddings for documents by encoding their text.

        :param docs: A document array with documents to create embeddings for. Only the
            documents that have the ``text`` attribute will get embeddings.
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
            needs_attr='text',
        )

        with torch.no_grad():
            for batch in batch_generator:
                embeddings = self.model.encode_text(text=[[doc.text] for doc in batch])
                embeddings = embeddings.cpu().numpy()

                for idx, doc in enumerate(batch):
                    doc.embedding = embeddings[idx]
