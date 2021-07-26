__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Union, Tuple, List, Iterable, Optional

import numpy as np
import torch
from jina import Executor, requests, DocumentArray
from jina_commons.batching import get_docs_batch_generator


class FlairTextEncoder(Executor):
    """
    Encode an array of string in size `B` into an ndarray in size `B x D`

    The ndarray potentially is BatchSize x (Channel x Height x Width)

    Internally, :class:`FlairTextEncoder` wraps the DocumentPoolEmbeddings from Flair.

    :param embeddings: the name of the embeddings. Supported models include
        - ``word:[ID]``: the classic word embedding model, the ``[ID]`` are listed at
        https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/CLASSIC_WORD_EMBEDDINGS.md
        - ``flair:[ID]``: the contextual embedding model, the ``[ID]`` are listed at
        https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/FLAIR_EMBEDDINGS.md
        - ``pooledflair:[ID]``: the pooled version of the contextual embedding model,
        the ``[ID]`` are listed at
        https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/FLAIR_EMBEDDINGS.md
        - ``byte-pair:[ID]``: the subword-level embedding model, the ``[ID]`` are listed at
        https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/BYTE_PAIR_EMBEDDINGS.md
        - ``Example``: ('word:glove', 'flair:news-forward', 'flair:news-backward')
    :param default_batch_size: size of each batch
    :param default_traversal_paths: traversal path of the Documents, (e.g. 'r', 'c')
    :param on_gpu: set to True if using GPU
    :param pooling_strategy: the strategy to merge the word embeddings into the chunk embedding.
    Supported strategies include ``mean``, ``min``, ``max``.
    """
    def __init__(self,
                 embeddings: Union[Tuple[str], List[str]] = ('word:glove',),
                 pooling_strategy: str = 'mean',
                 on_gpu: bool = False,
                 default_batch_size: int = 32,
                 default_traversal_paths: Optional[List[str]] = None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.embeddings = embeddings
        self.pooling_strategy = pooling_strategy
        self.max_length = -1  # reserved variable for future usages
        self.on_gpu = on_gpu
        self.default_batch_size = default_batch_size
        self.default_traversal_paths = default_traversal_paths or ['r']
        self._post_set_device = False
        self.device = torch.device('cuda:0') if self.on_gpu else torch.device('cpu')

        import flair
        flair.device = self.device
        embeddings_list = []
        for e in self.embeddings:
            model_name, model_id = e.split(':', maxsplit=1)
            emb = None
            try:
                if model_name == 'flair':
                    from flair.embeddings import FlairEmbeddings
                    emb = FlairEmbeddings(model_id)
                elif model_name == 'pooledflair':
                    from flair.embeddings import PooledFlairEmbeddings
                    emb = PooledFlairEmbeddings(model_id)
                elif model_name == 'word':
                    from flair.embeddings import WordEmbeddings
                    emb = WordEmbeddings(model_id)
                elif model_name == 'byte-pair':
                    from flair.embeddings import BytePairEmbeddings
                    emb = BytePairEmbeddings(model_id)
            except ValueError:
                # self.logger.error(f'embedding not found: {e}')
                continue
            if emb is not None:
                embeddings_list.append(emb)
        if embeddings_list:
            from flair.embeddings import DocumentPoolEmbeddings
            self.model = DocumentPoolEmbeddings(embeddings_list, pooling=self.pooling_strategy)
            # self.logger.info(f'flair encoder initialized with embeddings: {self.embeddings}')
        else:
            print('flair encoder initialization failed.')

    @requests
    def encode(self, docs: Optional[DocumentArray], parameters: dict, *args, **kwargs):
        """
        Encode text data into a ndarray of `D` as dimension, and fill the embedding of each Document.

        :param docs: documents sent to the encoder. The docs must have text.
        :param parameters: dictionary to define the `traversal_path` and the `batch_size`.
            For example,
            `parameters={'traversal_paths': ['r'], 'batch_size': 10}`
            will set the parameters for traversal_paths, batch_size and that are actually used
        """
        if docs:
            document_batches_generator = get_docs_batch_generator(
                docs,
                traversal_path=parameters.get('traversal_paths', self.default_traversal_paths),
                batch_size=parameters.get('batch_size', self.default_batch_size),
                needs_attr='text'
            )
            self._create_embeddings(document_batches_generator)

    def _create_embeddings(self, document_batches_generator: Iterable):
        for document_batch in document_batches_generator:
            from flair.data import Sentence
            c_batch = [Sentence(d.text) for d in document_batch]

            self.model.embed(c_batch)
            for document, c_text in zip(document_batch, c_batch):
                document.embedding = self.tensor2array(c_text.embedding)

    def tensor2array(self, tensor):
        if isinstance(tensor, np.ndarray):
            return tensor
        return tensor.cpu().numpy() if self.on_gpu else tensor.numpy()
