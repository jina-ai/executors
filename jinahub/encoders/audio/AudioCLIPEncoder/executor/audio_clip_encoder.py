__copyright__ = 'Copyright (c) 2020-2021 Jina AI Limited. All rights reserved.'
__license__ = 'Apache-2.0'

from typing import Any, Iterable, Optional

import librosa as lr
import numpy as np
import torch
from jina import DocumentArray, Executor, requests
from jina.excepts import BadDocType

from .audio_clip.model import AudioCLIP


class AudioCLIPEncoder(Executor):
    """
    Encode audio data with AudioCLIP embeddings
    """

    TARGET_SAMPLE_RATE = 44100  # derived from ESResNeXt

    def __init__(
        self,
        model_path: str = 'assets/AudioCLIP-Full-Training.pt',
        traversal_paths: Iterable[str] = ('r',),
        batch_size: int = 32,
        device: str = 'cpu',
        download_model: bool = True,
        *args,
        **kwargs
    ):
        """
        :param model_path: path of the pre-trained AudioCLIP model
        :param traversal_paths: default traversal path
        :param device: Torch device string (e.g. 'cpu', 'cuda', 'cuda:2')
        """

        super().__init__(*args, **kwargs)
        torch.set_grad_enabled(False)
        self.model_path = model_path
        self.traversal_paths = traversal_paths
        self.batch_size = batch_size

        if download_model:
            import os
            import subprocess

            root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            subprocess.call(['sh', 'scripts/download_model.sh'], cwd=root_path)

        try:
            self.model = AudioCLIP(pretrained=model_path).to(device).eval()
        except FileNotFoundError:
            raise FileNotFoundError(
                'Please download AudioCLIP model and set the `model_path` argument.'
            )

    @requests
    def encode(
        self,
        docs: Optional[DocumentArray] = None,
        parameters: dict = {},
        *args,
        **kwargs
    ) -> Any:
        """
        Encode all Documents with audio data (stored in the ``blob`` attribute) and store the
        embeddings in the ``embedding`` attribute of the Documents.

        :param docs: a `DocumentArray` contains `Document`s with `blob` of the size (n,) or (2, n).
            The `blob` contains audio time-series data. Additionally,
            `tags` of each `Document` must contain `sample_rate` field,
            which has the sample rate of the audio data. The `sample_rate` must be a positive
            scalar value.
        :param parameters: dictionary to defines the `traversal_paths`.
        """
        if not docs:
            return

        traversal_paths = parameters.get('traversal_paths', self.traversal_paths)
        batch_size = parameters.get('batch_size', self.batch_size)

        with torch.inference_mode():
            for batch in docs.batch(batch_size, traversal_paths):
                self._create_embeddings(batch)

    def _create_embeddings(self, filtered_docs: Iterable):
        """Update the documents with the embeddings generated by AudioCLIP"""

        for d in filtered_docs:
            d.blob, d.tags['sample_rate'] = self._resample(
                d.blob, d.tags.get('sample_rate', None)
            )
            audio = torch.Tensor(d.blob).unsqueeze(0)
            embedding = self.model.encode_audio(audio=audio)[0]
            d.embedding = embedding.cpu().numpy()

    def _resample(self, blob: np.ndarray, orig_sr: int):
        if orig_sr is None:
            raise BadDocType(
                'sample rate is not given, please provide a valid sample rate'
            )
        if orig_sr == AudioCLIPEncoder.TARGET_SAMPLE_RATE:
            return blob, orig_sr
        return (
            lr.resample(blob, orig_sr, AudioCLIPEncoder.TARGET_SAMPLE_RATE),
            AudioCLIPEncoder.TARGET_SAMPLE_RATE,
        )
