__copyright__ = 'Copyright (c) 2020-2021 Jina AI Limited. All rights reserved.'
__license__ = 'Apache-2.0'

from pathlib import Path
from typing import Optional, Dict

import torch
import librosa as lr
import soundfile as sf
from jina import Executor, DocumentArray, requests
from jina.excepts import BadDocType


class VADSpeechSegmenter(Executor):
    """
     Segment the speech audio using Silero's Voice Activity Detector (VAD)

    :param normalize: a bool to specify whether to normalize the audio
     by the sample rate
    :param dump: a bool to specify whether to dump the segmented audio
    :param repo: default repo name of silero-vad
    :param model: default model name of silero-vad

    """

    def __init__(
        self,
        normalize: bool = False,
        dump: bool = False,
        repo: str = 'snakers4/silero-vad',
        model: str = 'silero_vad',
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if model is None or repo is None:
            raise ValueError('model and repo cannot be None')

        # TODO: remove the following temporary fix for torch hub
        torch.hub._validate_not_a_forked_repo = lambda *args: True
        self.model, utils = torch.hub.load(
            repo_or_dir=repo, model=model, force_reload=True
        )

        (_, self.get_speech_ts_adaptive, self.save_audio, _, _, _, _) = utils
        self.normalize = self._normalize if normalize else lambda audio, _: audio
        self.dump = dump

    @requests
    def segment(
        self,
        docs: Optional['DocumentArray'] = None,
        parameters: Optional[Dict] = {},
        **kwargs,
    ):
        """
        Segment all audio docs to chunks using silero-vad
        :param docs: documents sent to the segmenter. The docs must have either
        `blob` or `uri` that ends with either `.mp3` or `.wav`.
        :param parameters: dictionary to override the default parameters.
        """

        for doc in docs:
            unnormalized_audio, sample_rate = self._load_audio(doc)
            audio = self.normalize(unnormalized_audio, sample_rate)
            speech_timestamps = self.get_speech_ts_adaptive(audio, self.model)
            doc.blob, doc.tags['sample_rate'] = (
                audio.detach().cpu().numpy(),
                sample_rate,
            )
            dump_dir = self.dump and (Path(self.workspace) / 'audio')

            if dump_dir:
                dump_dir.mkdir(exist_ok=True)
                self.save_audio(
                    str(dump_dir / f'doc_{doc.id}.wav'),
                    unnormalized_audio,
                    int(sample_rate),
                )

            chunk_path = f'{dump_dir}/chunk_{doc.id}_{{0}}_{{1}}.wav'
            for ts in speech_timestamps:
                doc.chunks.append(
                    Document(
                        blob=audio[ts['start'] : ts['end']].detach().cpu().numpy(),
                        location=[ts['start'], ts['end']],
                        tags=doc.tags,
                    )
                )

                if dump_dir:
                    self.save_audio(
                        chunk_path.format(ts['start'], ts['end']),
                        unnormalized_audio[ts['start'] : ts['end']],
                        int(sample_rate),
                    )

    def _load_audio(self, doc):
        if doc.blob is not None:
            return torch.Tensor(doc.blob), doc.tags.get('sample_rate')
        elif doc.uri is not None and doc.uri.endswith('.mp3'):
            audio, sample_rate = self._read_mp3(doc.uri)
            return torch.Tensor(audio), sample_rate
        elif doc.uri is not None and doc.uri.endswith('.wav'):
            audio, sample_rate = self._read_wav(doc.uri)
            return torch.Tensor(audio), sample_rate
        else:
            raise BadDocType('doc needs to have either a blob or a wav/mp3 uri')

    def _read_wav(self, file_path):
        data, sample_rate = sf.read(file_path, dtype='int16')
        if len(wav_data.shape) > 1:
            data = np.mean(wav_data, axis=1)
        return data, sample_rate

    def _read_mp3(self, file_path):
        return lr.load(file_path)

    def _normalize(self, data, sample_rate):
        if not sample_rate:
            return
        return data / sample_rate
