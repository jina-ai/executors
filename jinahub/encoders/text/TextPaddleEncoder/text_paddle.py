__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Dict, Iterable, Optional

import numpy as np
import paddlehub as hub
from jina import DocumentArray, Executor, requests


class TextPaddleEncoder(Executor):
    """
    Encode an array of string in size `B` into an ndarray in size `B x D`

    The ndarray potentially is BatchSize x (Channel x Height x Width)

    Internally, :class:`TextPaddlehubEncoder` wraps the Ernie module from paddlehub.
    https://github.com/PaddlePaddle/PaddleHub
    For models' details refer to
        https://www.paddlepaddle.org.cn/hublist?filter=en_category&value=SemanticModel
    """

    def __init__(
        self,
        model_name: Optional[str] = 'ernie_tiny',
        traversal_paths: Iterable[str] = ('r',),
        batch_size: int = 32,
        device: str = 'cpu',
        *args,
        **kwargs,
    ):
        """
        :param model_name: the name of the model. Supported models include
            ``ernie``, ``ernie_tiny``, ``ernie_v2_eng_base``, ``ernie_v2_eng_large``,
            ``bert_chinese_L-12_H-768_A-12``, ``bert_multi_cased_L-12_H-768_A-12``,
            ``bert_multi_uncased_L-12_H-768_A-12``, ``bert_uncased_L-12_H-768_A-12``,
            ``bert_uncased_L-24_H-1024_A-16``, ``chinese-bert-wwm``,
            ``chinese-bert-wwm-ext``, ``chinese-electra-base``,
            ``chinese-electra-small``, ``chinese-roberta-wwm-ext``,
            ``chinese-roberta-wwm-ext-large``, ``rbt3``, ``rbtl3``
        :param traversal_paths: fallback traversal path in case there is not traversal path sent in the request
        :param batch_size: fallback batch size in case there is not batch size sent in the request
        :param device: Device to be used. Use 'gpu' for GPU or use 'cpu' for CPU.
        """
        super().__init__(*args, **kwargs)
        self.device = device
        self.model = hub.Module(name=model_name)
        self.batch_size = batch_size
        self.traversal_paths = traversal_paths

    @requests
    def encode(
        self, docs: Optional[DocumentArray] = None, parameters: Dict = {}, **kwargs
    ):
        """Encode doc content into vector representation.

        :param docs: documents sent to the encoder. The docs must have the
            ``text`` attribute.
        :param parameters: dictionary to define the `traversal_paths` and the `batch_size`. For example,
            `parameters={'traversal_paths': ['r'], 'batch_size': 10}`.
        :param kwargs: Additional key value arguments.
        """
        if docs is None:
            return

        document_batches_generator = docs.batch(
            traversal_paths=parameters.get('traversal_paths', self.traversal_paths),
            batch_size=parameters.get('batch_size', self.batch_size),
            require_attr='text',
        )
        for batch in document_batches_generator:
            pooled_features = []
            results = self.model.get_embedding(
                batch.texts, use_gpu=self.device == 'gpu'
            )
            for pooled_feature, _ in results:
                pooled_features.append(pooled_feature)
            for doc, feature in zip(batch, pooled_features):
                doc.embedding = np.asarray(feature)
