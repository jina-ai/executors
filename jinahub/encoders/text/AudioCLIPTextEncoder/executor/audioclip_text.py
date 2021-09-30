__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Iterable, Optional

import torch
from jina import DocumentArray, Executor, requests

from .audio_clip.model import AudioCLIP


class AudioCLIPTextEncoder(Executor):
    """
    Encode text data with the AudioCLIP model
    """

    def __init__(
        self,
        model_path: str = '.cache/AudioCLIP-Full-Training.pt',
        tokenizer_path: str = '.cache/bpe_simple_vocab_16e6.txt.gz',
        traversal_paths: Iterable[str] = ('r',),
        batch_size: int = 32,
        device: str = 'cpu',
        *args,
        **kwargs
    ):
        """
        :param model_path: path to the pre-trained AudioCLIP model.
        :param traversal_paths: default traversal path (used if not specified in
            request's parameters)
        :param batch_size: default batch size (used if not specified in
            request's parameters)
        :param device: device that the model is on (should be "cpu", "cuda" or "cuda:X",
            where X is the index of the GPU on the machine)
        """
        super().__init__(*args, **kwargs)
        self.model = (
            AudioCLIP(
                pretrained=model_path,
                bpe_path=tokenizer_path,
            )
            .to(device)
            .eval()
        )
        self.traversal_paths = traversal_paths
        self.batch_size = batch_size

    @requests
    def encode(
        self,
        docs: Optional[DocumentArray] = None,
        parameters: dict = {},
        *args,
        **kwargs
    ) -> None:
        """
        Method to create embeddings for documents by encoding their text.

        :param docs: A document array with documents to create embeddings for. Only the
            documents that have the ``text`` attribute will get embeddings.
        :param parameters: A dictionary that contains parameters to control encoding.
            The accepted keys are ``traversal_paths`` and ``batch_size`` - in their
            absence their corresponding default values are used.
        """
        if not docs:
            return

        batch_generator = docs.batch(
            traversal_paths=parameters.get('traversal_paths', self.traversal_paths),
            batch_size=parameters.get('batch_size', self.batch_size),
            require_attr='text',
        )

        with torch.inference_mode():
            for batch in batch_generator:
                embeddings = self.model.encode_text(text=[[doc.text] for doc in batch])
                embeddings = embeddings.cpu().numpy()

                for idx, doc in enumerate(batch):
                    doc.embedding = embeddings[idx]
