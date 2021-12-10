__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Iterable, Optional, Sequence

import flair
import torch
from flair.data import Sentence
from flair.embeddings import (
    BytePairEmbeddings,
    DocumentPoolEmbeddings,
    FlairEmbeddings,
    WordEmbeddings,
)
from jina import DocumentArray, Executor, requests


class FlairTextEncoder(Executor):
    """Encode text into embeddings using models from the flair library.

    This module provides a subset sentence embedding functionality from the flair
    library, namely it allows you classical word embeddings, byte-pair embeddings and
    flair embeddings, and create sentence embeddings from a combtination of these models
    using document pool embeddings.

    Due to different interfaces of all these embedding models, using custom pre-trained
    models (not part of the library), or other embedding models is not possible. For
    that, we recommend that you create a custom executor.
    """

    def __init__(
        self,
        embeddings: Sequence[str] = ('word:glove',),
        pooling_strategy: str = 'mean',
        traversal_paths: Iterable[str] = ('r',),
        batch_size: int = 32,
        device: str = 'cpu',
        *args,
        **kwargs,
    ):
        """
        :param embeddings: the name of the embeddings. Supported models include
            - ``word:[ID]``: the classic word embedding model, the ``[ID]`` are listed at
            https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/CLASSIC_WORD_EMBEDDINGS.md
            - ``flair:[ID]``: the contextual embedding model, the ``[ID]`` are listed at
            https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/FLAIR_EMBEDDINGS.md
            - ``byte-pair:[ID]``: the subword-level embedding model, the ``[ID]`` are listed at
            https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/BYTE_PAIR_EMBEDDINGS.md

            Example: ``('word:glove', 'flair:news-forward', 'flair:news-backward')``
        :param pooling_strategy: the strategy to merge the word embeddings into the sentence
            embedding. Supported strategies are ``'mean'``, ``'min'`` and ``'max'``.
        :param traversal_paths: Default traversal paths, used if ``traversal_paths``
            are not provided as a parameter in the request.
        :param batch_size: Default batch size, used if ``batch_size`` is not
            provided as a parameter in the request
        :param device: The device (cpu or gpu) that the model should be on.
        """
        super().__init__(*args, **kwargs)

        if isinstance(embeddings, str):
            raise ValueError(
                'embeddings can not be a string, you need to pass a tuple or a list'
            )

        if pooling_strategy not in ['min', 'max', 'mean']:
            raise ValueError(
                'pooling_strategy has to be one of "min", "max" or "mean", got'
                f'{pooling_strategy}'
            )

        self.pooling_strategy = pooling_strategy
        self.batch_size = batch_size
        self.traversal_paths = traversal_paths
        self.device = torch.device(device)

        flair.device = self.device
        embeddings_list = []
        for e in embeddings:
            model_name, model_id = e.split(':', maxsplit=1)
            model_dict = {
                'flair': FlairEmbeddings,
                'word': WordEmbeddings,
                'byte-pair': BytePairEmbeddings,
            }

            try:
                model_class = model_dict[model_name]
            except KeyError:
                raise ValueError(
                    f'The model name {model_name} not recognized, valid model names'
                    ' are "flair", "word" and "byte-pair"'
                )
            embeddings_list.append(model_class(model_id))

        self.model = DocumentPoolEmbeddings(
            embeddings_list, pooling=self.pooling_strategy
        )

    @requests
    def encode(
        self, docs: Optional[DocumentArray] = None, parameters: dict = {}, **kwargs
    ):
        """
        Encode text data into a ndarray of `D` as dimension, and fill the embedding of each Document.

        :param docs: documents sent to the encoder. The docs must have `text`.
        :param parameters: dictionary to define the `traversal_path` and the `batch_size`.
            For example,
            `parameters={'traversal_paths': ['r'], 'batch_size': 10}`
            will set the parameters for traversal_paths, batch_size and that are actually used
        """
        if docs is None:
            return

        document_batches_generator = docs.traverse_flat(
            traversal_paths=parameters.get('traversal_paths', self.traversal_paths),
            filter_fn=lambda doc: len(doc.text) > 0,
        ).batch(
            batch_size=parameters.get('batch_size', self.batch_size),
        )

        for document_batch in document_batches_generator:

            c_batch = [Sentence(d.text) for d in document_batch]

            self.model.embed(c_batch)
            for document, c_text in zip(document_batch, c_batch):
                document.embedding = c_text.embedding.cpu().numpy()
