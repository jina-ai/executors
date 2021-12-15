__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Dict, Iterable, Optional

import torch
from jina import DocumentArray, Executor, requests
from sentence_transformers import SentenceTransformer


class TransformerSentenceEncoder(Executor):
    """
    Encode the Document text into embedding.
    """

    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        traversal_paths: Iterable[str] = ('r',),
        batch_size: int = 32,
        device: str = 'cpu',
        *args,
        **kwargs
    ):
        """
        :param model_name: The name of the sentence transformer to be used
        :param device: Torch device to put the model on (e.g. 'cpu', 'cuda', 'cuda:1')
        :param traversal_paths: Default traversal paths
        :param batch_size: Batch size to be used in the encoder model
        """
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.traversal_paths = traversal_paths
        self.model = SentenceTransformer(model_name, device=device)

    @requests
    def encode(
        self, docs: Optional[DocumentArray] = None, parameters: Dict = {}, **kwargs
    ):
        """
        Encode all docs with text and store the encodings in the ``embedding`` attribute
        of the docs.

        :param docs: Documents to send to the encoder. They need to have the ``text``
            attribute get an embedding.
        :param parameters: Any additional parameters for the `encode` function.
        """
        if docs is None:
            return

        for batch in docs.traverse_flat(
            traversal_paths=parameters.get('traversal_paths', self.traversal_paths),
            filter_fn=lambda doc: len(doc.text) > 0,
        ).batch(
            batch_size=parameters.get('batch_size', self.batch_size),
        ):
            texts = batch.get_attributes('text')

            with torch.inference_mode():
                embeddings = self.model.encode(texts)
                for doc, embedding in zip(batch, embeddings):
                    doc.embedding = embedding
