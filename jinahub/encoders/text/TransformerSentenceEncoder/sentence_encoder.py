__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Optional, Dict, List, Tuple

from jina import Executor, DocumentArray, requests
from sentence_transformers import SentenceTransformer
from jina_commons.batching import get_docs_batch_generator


class TransformerSentenceEncoder(Executor):
    """
    Encode the Document text into embedding.

    :param embedding_dim: the output dimensionality of the embedding
    """

    def __init__(
        self,
        model_name: str = 'sentence-transformers/paraphrase-mpnet-base-v2',
        device: str = "cpu",
        default_traversal_paths: Tuple = ('r', ),
        default_batch_size=32,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.default_batch_size = default_batch_size
        self.default_traversal_paths = default_traversal_paths
        self.model = SentenceTransformer(model_name, device=device)

    @requests
    def encode(self, docs: Optional[DocumentArray], parameters: Dict, **kwargs):
        """
        Encode all docs with images and store the encodings in the embedding attribute of the docs.
        :param docs: documents sent to the encoder. The docs must have `blob` of the shape `256`.
        :param parameters: Any additional parameters for the `encode` function.
        """
        for batch in get_docs_batch_generator(
                docs,
                traversal_path=parameters.get('traversal_paths', self.default_traversal_paths),
                batch_size=parameters.get('batch_size', self.default_batch_size),
                needs_attr='text'
        ):
            texts = batch.get_attributes("text")
            embeddings = self.model.encode(texts)
            for doc, embedding in zip(batch, embeddings):
                doc.embedding = embedding
