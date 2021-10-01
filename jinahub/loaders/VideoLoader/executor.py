__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"


import glob
import io
import os
import random
import shutil
import string
import subprocess
import urllib.request
from typing import Optional

import librosa
import moviepy.editor as mp
import numpy as np
from jina import Document, DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger
from jina.types.document import _is_datauri

DEFAULT_FPS = 1


class VideoLoader(Executor):
    """
    Extract the frames from videos with `ffmpeg`
    """

    def __init__(
        self, max_num_frames: int = 50, fps=DEFAULT_FPS, debug=False, **kwargs
    ):
        """

        :param max_num_frames:
        :param fps:
        :param debug: If True, the extracted frames are kept in `{workspace}/{video_fn}/*.jpg`
        :param type_extractor: What mime_type to be extracted from video data
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.max_num_frames = max_num_frames
        self.fps = fps
        self.logger = JinaLogger(
            getattr(self.metas, 'name', self.__class__.__name__)
        ).logger
        self.debug = debug

    @requests
    def extract(self, docs: Optional[DocumentArray] = None, **kwargs):
        """
        Load the video from the Document.uri, extract frames and save the frames into chunks.blob

        :param docs: the input Documents with either the video file name or URL in the `uri` field
        """
        if not docs:
            return

        for doc in docs:
            self.logger.info(f'received {doc.id}')

            if doc.uri == '':
                raise ValueError(f'No uri passed along with the Document for {doc.id}')

            # extract all the frames and audio per video
            frame_fn_list, audio_file = self._extract(doc.uri)

            # add frames as chunks to the Document, with modality='image'
            for frame_fn in frame_fn_list:
                self.logger.debug(f'frame: {frame_fn}')
                _chunk = Document(uri=frame_fn, modality='image')
                _chunk.convert_uri_to_datauri()
                _chunk.convert_image_datauri_to_blob()
                _chunk.blob = np.array(_chunk.blob).astype('uint8')
                timestamp = self._get_timestamp_from_filename(frame_fn)
                _chunk.location.append(np.uint32(timestamp))
                # _chunk.uri = frame_fn
                doc.chunks.append(_chunk)

            # add audio as chunks too to the same Document but with modality='audio'
            _chunk = Document(uri=audio_file, modality='audio')
            _chunk.convert_uri_to_datauri()
            _chunk.blob, _chunk.tags['sample_rate'] = librosa.load(audio_file)
            # _chunk.uri = frame_fn
            doc.chunks.append(_chunk)

            if not self.debug:
                frame_fn_list.append(audio_file)
                self._delete_tmp(frame_fn_list)

    def _extract(self, uri):
        source_fn = self._save_uri_to_tmp_file(uri) if _is_datauri(uri) else uri
        self.logger.debug(f'extracting {source_fn}')
        _base_fn = os.path.basename(uri).split('.')[0]
        target_path_frames = os.path.join(self.workspace, f'{_base_fn}', 'images')
        target_path_audio = os.path.join(self.workspace, f'{_base_fn}', 'audio')
        result = []
        os.makedirs(target_path_frames, exist_ok=True)
        os.makedirs(target_path_audio, exist_ok=True)
        try:
            subprocess.check_call(
                f'ffmpeg -loglevel panic -i {source_fn} -vsync 0 -vf fps={self.fps} -frame_pts true -s 960x540 '
                f'{os.path.join(target_path_frames, f"%d.jpg")} 2>&1',
                shell=True,
            )
            subprocess.check_call(
                f'ffmpeg -loglevel panic -i {source_fn} -ab 160k -ac 2 -ar 44100 -vn '
                f'{os.path.join(target_path_audio, f"audio.wav")} 2>&1',
                shell=True,
            )
        except subprocess.CalledProcessError as e:
            self.logger.error(f'Extraction failed, {uri}, {e}')
        finally:
            for fn in glob.glob(f'{target_path_frames}/*.jpg'):
                result.append(fn)
            if _is_datauri(uri):
                os.remove(source_fn)
            return result[: self.max_num_frames], os.path.join(
                target_path_audio, 'audio.wav'
            )

    def _save_uri_to_tmp_file(self, uri):
        req = urllib.request.Request(uri, headers={'User-Agent': 'Mozilla/5.0'})
        tmp_fn = os.path.join(
            self.workspace,
            ''.join([random.choice(string.ascii_lowercase) for i in range(10)])
            + '.mp4',
        )
        with urllib.request.urlopen(req) as fp:
            buffer = fp.read()
            binary_fn = io.BytesIO(buffer)
            with open(tmp_fn, 'wb') as f:
                f.write(binary_fn.read())
        return tmp_fn

    def _delete_tmp(self, frame_fn_list):
        _path_to_remove = set()
        for fn in frame_fn_list:
            if os.path.exists(fn):
                _path = os.path.dirname(fn)
                _path_to_remove.add(_path)
        for _path in _path_to_remove:
            try:
                shutil.rmtree(_path)
            except OSError as e:
                self.logger.error(f'Error in deleting {_path}: {e}')

    def convert_uri_to_audio_blob(self, uri):
        my_clip = mp.VideoFileClip(uri)
        my_clip.audio.write_audiofile('tests/toy_data/abc.mp3')

    def _get_timestamp_from_filename(self, uri):
        return os.path.basename(uri).split('.')[0]
