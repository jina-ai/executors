__copyright__ = 'Copyright (c) 2020-2021 Jina AI Limited. All rights reserved.'
__license__ = 'Apache-2.0'

from pathlib import Path
from typing import Optional, Tuple

import librosa as lr
import numpy as np
import torch
import torchaudio
from jina import Document, DocumentArray, Executor, requests
from jina.excepts import BadDocType


class VADSpeechSegmenter(Executor):
    """
     Segment the speech audio using Silero's Voice Activity Detector (VAD)
    """

    TARGET_SAMPLE_RATE = 16000

    def __init__(
        self,
        normalize: bool = False,
        dump: bool = False,
        *args,
        **kwargs,
    ):
        """
        Segment the speech audio using Silero's Voice Activity Detector (VAD)
        :param normalize: a bool to specify whether to normalize the audio
         by the sample rate
        :param dump: a bool to specify whether to dump the segmented audio
        """
        super().__init__(*args, **kwargs)
        # TODO: remove the following temporary fix for torch hub
        torch.hub._validate_not_a_forked_repo = lambda *args: True
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True
        )
        self.model.eval()  # set model to eval mode

        (_, self.get_speech_ts_adaptive, self.save_audio, *_) = utils
        self.normalize = self._normalize if normalize else lambda audio, _: audio
        self.dump = dump

    @requests
    def segment(
        self,
        docs: Optional[DocumentArray] = None,
        parameters: dict = {},
        **kwargs,
    ):
        """
        Segment all audio docs to chunks using silero-vad
        :param docs: documents sent to the segmenter. The docs must have either
        `blob` or `uri` that ends with either `.mp3` or `.wav`.
        :param parameters: dictionary to override the default parameters.
        """
        if not docs:
            return

        for doc in docs:
            unnormalized_audio, sample_rate = self._resample(*self._load_raw_audio(doc))
            audio = self.normalize(unnormalized_audio, sample_rate)
            speech_timestamps = self.get_speech_ts_adaptive(audio, self.model)
            doc.blob, doc.tags['sample_rate'] = (
                audio.cpu().numpy(),
                sample_rate,
            )

            if self.dump:
                dump_dir = Path(self.workspace) / 'audio'
                dump_dir.mkdir(exist_ok=True)
                self.save_audio(
                    str(dump_dir / f'doc_{doc.id}_original.wav'),
                    unnormalized_audio,
                    sample_rate,
                )
                chunk_path = f'{dump_dir}/chunk_{doc.id}_{{0}}_{{1}}.wav'

            for ts in speech_timestamps:
                doc.chunks.append(
                    Document(
                        blob=audio[ts['start'] : ts['end']].cpu().numpy(),
                        location=[ts['start'], ts['end']],
                        tags=doc.tags,
                    )
                )

                if self.dump:
                    self.save_audio(
                        chunk_path.format(ts['start'], ts['end']),
                        unnormalized_audio[ts['start'] : ts['end']],
                        sample_rate,
                    )

    def _load_raw_audio(self, doc: Document) -> Tuple[torch.Tensor, int]:
        if doc.blob is not None and doc.tags.get('sample_rate', None) is None:
            raise BadDocType('data is blob but sample rate is not provided')
        elif doc.blob is not None:
            return torch.Tensor(doc.blob), int(doc.tags['sample_rate'])
        elif doc.uri is not None and doc.uri.endswith('.mp3'):
            audio, sample_rate = self._read_mp3(doc.uri)
            return torch.Tensor(audio), int(sample_rate)
        elif doc.uri is not None and doc.uri.endswith('.wav'):
            audio, sample_rate = self._read_wav(doc.uri)
            return torch.Tensor(audio), int(sample_rate)
        else:
            raise BadDocType('doc needs to have either a blob or a wav/mp3 uri')

    def _read_wav(self, file_path: str) -> Tuple[np.ndarray, int]:
        data, sample_rate = torchaudio.load(file_path)
        data = np.mean(data.cpu().numpy(), axis=0)
        return data, sample_rate

    def _read_mp3(self, file_path: str) -> Tuple[np.ndarray, int]:
        return lr.load(file_path)

    def _normalize(self, data: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if sample_rate == 0:
            raise BadDocType('sample rate cannot be 0')
        return data / sample_rate

    def _resample(
        self,
        data: torch.Tensor,
        orig_sample_rate: int,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
    ) -> Tuple[torch.Tensor, int]:

        if orig_sample_rate == 0:
            raise BadDocType('sample rate cannot be 0')

        if orig_sample_rate == target_sample_rate:
            return data, orig_sample_rate

        transform = torchaudio.transforms.Resample(
            orig_freq=orig_sample_rate, new_freq=target_sample_rate
        )
        return transform(data), target_sample_rate
