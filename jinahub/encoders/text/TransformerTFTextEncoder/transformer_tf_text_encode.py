__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Any, Iterable, List, Optional

import numpy as np
import tensorflow as tf
from jina import DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger


def _batch_generator(data: List[Any], batch_size: int):
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


class TransformerTFTextEncoder(Executor):
    """
    Internally wraps the tensorflow-version of transformers from huggingface.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = 'distilbert-base-uncased',
        base_tokenizer_model: Optional[str] = None,
        pooling_strategy: str = 'mean',
        layer_index: int = -1,
        max_length: Optional[int] = None,
        default_batch_size: int = 32,
        default_traversal_paths: List[str] = None,
        on_gpu: bool = False,
        *args,
        **kwargs,
    ):
        """
        :param pretrained_model_name_or_path: Either:
            - a string, the `model id` of a pretrained model hosted inside a
                model repo on huggingface.co, e.g.: ``bert-base-uncased``.
            - a path to a `directory` containing model weights saved using
                :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.:
                ``./my_model_directory/``.
        :param base_tokenizer_model: The name of the base model to use for
            creating the tokenizer. If None, will be equal to
            `pretrained_model_name_or_path`.
        :param pooling_strategy: the strategy to merge the word embeddings
            into the chunk embedding. Supported strategies include
            'cls', 'mean', 'max', 'min'.
        :param layer_index: index of the transformer layer that is used to
            create encodings. Layer 0 corresponds to the embeddings layer
        :param max_length: the max length to truncate the tokenized sequences to.
        :param default_batch_size: size of each batch
        :param default_traversal_paths: traversal path of the Documents, (e.g. 'r', 'c')
        :param on_gpu: set to True if using GPU
        """
        super().__init__(*args, **kwargs)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.base_tokenizer_model = (
            base_tokenizer_model or pretrained_model_name_or_path
        )
        self.pooling_strategy = pooling_strategy
        self.layer_index = layer_index
        self.max_length = max_length
        self.default_batch_size = default_batch_size
        self.default_traversal_paths = default_traversal_paths or ['r']
        self.on_gpu = on_gpu
        self.logger = JinaLogger(self.__class__.__name__)

        if self.pooling_strategy == 'auto':
            self.pooling_strategy = 'cls'
            self.logger.warning(
                '"auto" pooling_strategy is deprecated, Defaulting to '
                ' "cls" to maintain the old default behavior.'
            )

        if self.pooling_strategy not in ['cls', 'mean', 'max', 'min']:
            self.logger.error(
                f'pooling strategy not found: {self.pooling_strategy}.'
                ' The allowed pooling strategies are "cls", "mean", "max", "min".'
            )
            raise NotImplementedError

        from transformers import AutoTokenizer, TFAutoModel

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_tokenizer_model)
        self.model = TFAutoModel.from_pretrained(
            self.pretrained_model_name_or_path, output_hidden_states=True
        )

        import tensorflow as tf

        cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        if self.on_gpu and len(gpus) > 0:
            cpus.append(gpus[0])
        if self.on_gpu and len(gpus) == 0:
            self.logger.warning(
                'You tried to use a GPU but no GPU was found on'
                ' your system. Defaulting to CPU!'
            )
        tf.config.experimental.set_visible_devices(devices=cpus)

    @requests
    def encode(self, docs: DocumentArray, parameters: dict, *args, **kwargs):
        """
        Encode an array of string in size `B` into an ndarray in size `B x D`,
        where `B` is the batch size and `D` is the dimensionality of the encoding.
        :param docs: DocumentArray containing images
        :param parameters: dictionary parameters
        """
        if docs:
            document_batches_generator = self._get_input_data(docs, parameters)
            self._create_embeddings(document_batches_generator)

    def _get_embeddings(self, text):
        if not self.tokenizer.pad_token:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer.vocab))

        input_tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='longest',
            truncation=True,
            return_tensors='tf',
        )

        outputs = self.model(**input_tokens)

        n_layers = len(outputs.hidden_states)
        if self.layer_index not in list(range(-n_layers, n_layers)):
            self.logger.error(
                f'Invalid value {self.layer_index} for `layer_index`,'
                f' for the model {self.pretrained_model_name_or_path}'
                f' valid values are integers from {-n_layers} to {n_layers - 1}.'
            )
            raise ValueError

        if self.pooling_strategy == 'cls' and not self.tokenizer.cls_token:
            self.logger.error(
                f'You have set pooling_strategy to "cls", but the tokenizer'
                f' for the model {self.pretrained_model_name_or_path}'
                f' does not have a cls token set.'
            )
            raise ValueError

        fill_vals = {'cls': 0.0, 'mean': 0.0, 'max': -np.inf, 'min': np.inf}
        fill_val = tf.constant(fill_vals[self.pooling_strategy])

        layer = outputs.hidden_states[self.layer_index]
        attn_mask = tf.expand_dims(input_tokens['attention_mask'], -1)
        attn_mask = tf.broadcast_to(attn_mask, layer.shape)
        layer = tf.where(attn_mask == 1, layer, fill_val)

        if self.pooling_strategy == 'cls':
            CLS = self.tokenizer.cls_token_id
            ind = tf.experimental.numpy.nonzero(input_tokens['input_ids'] == CLS)
            output = tf.gather_nd(layer, tf.stack(ind, axis=1))
        elif self.pooling_strategy == 'mean':
            output = tf.reduce_sum(layer, 1) / tf.reduce_sum(
                tf.cast(attn_mask, tf.float32), 1
            )
        elif self.pooling_strategy == 'max':
            output = tf.reduce_max(layer, 1)
        elif self.pooling_strategy == 'min':
            output = tf.reduce_min(layer, 1)

        return output.numpy()

    def _create_embeddings(self, document_batches_generator: Iterable):
        for document_batch in document_batches_generator:
            text_batch = [d.text for d in document_batch]
            embeddings = self._get_embeddings(text_batch)

            for document, embedding in zip(document_batch, embeddings):
                document.embedding = embedding

    def _get_input_data(self, docs: DocumentArray, parameters: dict):
        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )
        batch_size = parameters.get('batch_size', self.default_batch_size)

        # traverse thought all documents which have to be processed
        flat_docs = docs.traverse_flat(traversal_paths)

        # filter out documents without images
        filtered_docs = [doc for doc in flat_docs if doc.text is not None]

        return _batch_generator(filtered_docs, batch_size)
