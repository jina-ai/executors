__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
from pathlib import Path
from typing import Any, Iterable, List, Optional

import requests as _requests
import tensorflow as tf
from jina import DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger

tf.compat.v1.disable_eager_execution()

from .vggish.vggish_postprocess import *
from .vggish.vggish_slim import *

cur_dir = os.path.dirname(os.path.abspath(__file__))


class VggishAudioEncoder(Executor):
    """
    Encode audio data with Vggish embeddings

    :param model_path: path of the models directory
    :param default_traversal_paths: fallback batch size in case there is not batch size sent in the request
    :param device: device to run the model on e.g. 'cpu'/'cuda'/'cuda:2'
    """

    def __init__(
        self,
        model_path: str = Path(cur_dir) / 'models',
        default_traversal_paths: Optional[Iterable[str]] = None,
        device: str = 'cpu',
        *args,
        **kwargs
    ):

        super().__init__(*args, **kwargs)
        self.default_traversal_paths = default_traversal_paths or ['r']
        self.logger = JinaLogger(self.__class__.__name__)
        self.device = device
        self.model_path = Path(model_path)
        self.vgg_model_path = self.model_path / 'vggish_model.ckpt'
        self.pca_model_path = self.model_path / 'vggish_pca_params.ckpt'
        self.model_path.mkdir(
            exist_ok=True
        )  # Create the model directory if it does not exist yet

        cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        if 'cuda' in device and len(gpus) > 0:
            gpu_index = 0 if 'cuda:' not in device else int(device.split(':')[1])
            cpus.append(gpus[gpu_index])
        if 'cuda' in self.device and len(gpus) == 0:
            self.logger.warning(
                'You tried to use a GPU but no GPU was found on'
                ' your system. Defaulting to CPU!'
            )
        tf.config.experimental.set_visible_devices(devices=cpus)

        if not self.vgg_model_path.exists():
            self.logger.info(
                'VGGish model cannot be found from the given model path, downloading a new one...'
            )
            try:
                r = _requests.get(
                    'https://storage.googleapis.com/audioset/vggish_model.ckpt'
                )
                r.raise_for_status()
            except _requests.exceptions.HTTPError:
                self.logger.error(
                    'received HTTP error response, cannot download vggish model'
                )
                raise
            except _requests.exceptions.RequestException:
                self.logger.error('Connection error, cannot download vggish model')
                raise

            with open(self.vgg_model_path, 'wb') as f:
                f.write(r.content)

        if not self.pca_model_path.exists():
            self.logger.info(
                'PCA model cannot be found from the given model path, downloading a new one...'
            )
            try:
                r = _requests.get(
                    'https://storage.googleapis.com/audioset/vggish_pca_params.npz'
                )
                r.raise_for_status()
            except _requests.exceptions.HTTPError:
                self.logger.error(
                    'received HTTP error response, cannot download pca model'
                )
                raise
            except _requests.exceptions.RequestException:
                self.logger.error('Connection error, cannot download pca model')
                raise

            with open(self.pca_model_path, 'wb') as f:
                f.write(r.content)

        self.sess = tf.compat.v1.Session()
        define_vggish_slim()
        load_vggish_slim_checkpoint(self.sess, str(self.vgg_model_path))
        self.feature_tensor = self.sess.graph.get_tensor_by_name(INPUT_TENSOR_NAME)
        self.embedding_tensor = self.sess.graph.get_tensor_by_name(OUTPUT_TENSOR_NAME)
        self.post_processor = Postprocessor(str(self.pca_model_path))

    @requests
    def encode(self, docs: Optional[DocumentArray], parameters: dict, **kwargs):
        """
        Compute embeddings and store them in the `docs` array.

        :param docs: documents sent to the encoder. The docs must have `text`.
            By default, the input `text` must be a `list` of `str`.
        :param parameters: dictionary to define the `traversal_paths` and the `batch_size`. For example,
               `parameters={'traversal_paths': ['r'], 'batch_size': 10}`.
        :param kwargs: Additional key value arguments.
        :return:
        """
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

        # filter out documents without images
        filtered_docs = DocumentArray(
            [doc for doc in flat_docs if doc.blob is not None]
        )

        return filtered_docs

    def _create_embeddings(self, filtered_docs: Iterable):
        """Update the documents with the embeddings generated by VGGISH"""

        for d in filtered_docs:
            # Vggish broadcasts across different length audios, not batches
            [embedding] = self.sess.run(
                [self.embedding_tensor], feed_dict={self.feature_tensor: d.blob}
            )
            result = self.post_processor.postprocess(embedding)
            d.embedding = np.mean((np.float32(result) - 128.0) / 128.0, axis=0)

    def close(self):
        self.sess.close()
