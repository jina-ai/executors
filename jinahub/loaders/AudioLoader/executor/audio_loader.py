from typing import Iterable
from jina import Executor, DocumentArray, requests


class AudioLoader(Executor):
    """AudioLoader loads audio file into the Document buffer."""

    def __init__(self,
                 audio_types: Iterable[str] = None,
                 **kwargs):
        """
        Initializer function for AudioLoader executor
        Args:
            audio_types: List of strings of audio types that are allowed
                Supported types are 'mp3' and 'wav'.
            **kwargs: Keyword arguments
        """
        super().__init__(**kwargs)
        self.audio_types = audio_types or ['mp3', 'wav']
        for audio_type in self.audio_types:
            if audio_type.lower() not in ['mp3', 'wav']:
                raise ValueError(f'Audio Type "{audio_type}" not supported!')

    @requests
    def load_audio(self, docs: DocumentArray, **kwargs):
        pass
