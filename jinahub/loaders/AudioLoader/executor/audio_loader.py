from typing import Dict, Iterable

import librosa
from jina import DocumentArray, Executor, requests


class AudioLoader(Executor):
    """AudioLoader loads audio file into the Document buffer."""

    def __init__(
        self,
        audio_types: Iterable[str] = None,
        target_sample_rate: int = 22050,
        traversal_paths: Iterable[str] = None,
        **kwargs,
    ):
        """
        Initializer function for AudioLoader executor
        Args:
            audio_types: List of strings of audio types that are allowed
                Supported types are 'mp3' and 'wav'.
            target_sample_rate: sample rate that librosa converts the audio to.
                Default is 22050 (librosa's default value)
            **kwargs: Keyword arguments
        """
        super().__init__(**kwargs)
        self.audio_types = audio_types or ['mp3', 'wav']
        self.audio_types = [fformat.lower() for fformat in self.audio_types]
        self.audio_mime_types = {
            'mp3': ['audio/mpeg'],
            'wav': ['audio/x-wav', 'audio/wav'],
        }
        self.target_sample_rate = target_sample_rate
        self.traversal_paths = traversal_paths or ['r']
        for audio_type in self.audio_types:
            if audio_type not in ['mp3', 'wav']:
                raise ValueError(f'Audio Type "{audio_type}" not supported!')

    @requests
    def load_audio(self, docs: DocumentArray, parameters: Dict, **kwargs):
        traversal_paths = parameters.get('traversal_paths', self.traversal_paths)
        flat_docs = docs.traverse_flat(traversal_paths)

        for audio_type in self.audio_types:
            type_docs = DocumentArray(
                [
                    doc
                    for doc in flat_docs
                    if doc.mime_type in self.audio_mime_types[audio_type]
                    and doc.uri is not None
                ]
            )
            self.read_audio(type_docs)

    def read_audio(self, docs: DocumentArray):
        for doc in docs:
            blob, sample_rate = librosa.load(doc.uri, sr=self.target_sample_rate)
            doc.blob = blob
            doc.tags['sample_rate'] = sample_rate
