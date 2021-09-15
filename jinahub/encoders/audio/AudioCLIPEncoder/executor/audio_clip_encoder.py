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
    :param model_path: path of the pre-trained AudioCLIP model
    :param default_traversal_paths: default traversal path
    :param device: Torch device string (e.g. 'cpu', 'cuda', 'cuda:2')
    """

    TARGET_SAMPLE_RATE = 44100  # derived from ESResNeXt

    def __init__(
        self,
        model_path: str = 'assets/AudioCLIP-Full-Training.pt',
        default_traversal_paths: Iterable[str] = ('r',),
        device: str = 'cpu',
        *args,
        **kwargs
    ):

        super().__init__(*args, **kwargs)
        torch.set_grad_enabled(False)
        self.model_path = model_path
        self.aclp = AudioCLIP(pretrained=model_path).to(device).eval()
        self.default_traversal_paths = default_traversal_paths

    @requests
    def encode(
        self, docs: Optional[DocumentArray], parameters: dict, *args, **kwargs
    ) -> Any:

        if docs:
            cleaned_document_array = self._get_input_data(docs, parameters)
            self._create_embeddings(cleaned_document_array)

    def _get_input_data(self, docs: DocumentArray, parameters: dict):
        """Create a filtered set of Documents to iterate over."""

        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )

        # traverse thought all documents which have to be processed
        flat_docs = docs.traverse_flat(traversal_paths)

        # filter out documents without audio wav
        filtered_docs = DocumentArray(
            [doc for doc in flat_docs if doc.blob is not None]
        )

        return filtered_docs

    def _create_embeddings(self, filtered_docs: Iterable):
        """Update the documents with the embeddings generated by AudioCLIP"""

        for d in filtered_docs:
            d.blob, d.tags['sample_rate'] = self._resample(
                d.blob, d.tags.get('sample_rate', None)
            )
            audio = torch.Tensor(d.blob).unsqueeze(0)
            embedding = self.aclp.encode_audio(audio=audio)[0]
            d.embedding = embedding.cpu().numpy()

    def _resample(self, blob: np.ndarray, orig_sr: int):
        if orig_sr is None:
            raise BadDocType(
                'sample rate is not given, please provide a valid sample rate'
            )
        if orig_sr == AudioCLIPEncoder.TARGET_SAMPLE_RATE:
            return
        return (
            lr.resample(blob, orig_sr, AudioCLIPEncoder.TARGET_SAMPLE_RATE),
            AudioCLIPEncoder.TARGET_SAMPLE_RATE,
        )