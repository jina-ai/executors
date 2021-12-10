import os
import pickle
from pathlib import Path
from typing import Optional, Tuple

from jina import DocumentArray, Executor, requests
from jina.excepts import ExecutorFailToLoad


class TFIDFTextEncoder(Executor):
    """
    Encode text into tf-idf sparse embeddings
    """

    def __init__(
        self,
        path_vectorizer: Optional[str] = None,
        batch_size: int = 2048,
        traversal_paths: Tuple[str] = ('r',),
        *args,
        **kwargs,
    ):
        """
        :param path_vectorizer: path of the pre-trained tfidf sklearn vectorizer
        :param traversal_paths: fallback traversal path in case there is not traversal path sent in the request
        :param batch_size: fallback batch size in case there is not batch size sent in the request
        """
        super().__init__(*args, **kwargs)
        if path_vectorizer is None:
            path_vectorizer = str(
                Path(__file__).parent / 'model/tfidf_vectorizer.pickle'
            )

        self.path_vectorizer = path_vectorizer
        self.batch_size = batch_size
        self.traversal_paths = traversal_paths

        if os.path.exists(self.path_vectorizer):
            self.tfidf_vectorizer = pickle.load(open(self.path_vectorizer, 'rb'))
        else:
            raise ExecutorFailToLoad(
                f'{self.path_vectorizer} not found, cannot find a fitted tfidf_vectorizer'
            )

    @requests
    def encode(
        self, docs: Optional[DocumentArray] = None, parameters: dict = {}, **kwargs
    ):
        """
        Generate the TF-IDF feature vector for all text documents.

        :param docs: documents sent to the encoder. The docs must have `text`.
            By default, the input `text` must be a `list` of `str`.
        :param parameters: dictionary to define the `traversal_paths` and the
            `batch_size`. For example,
            `parameters={'traversal_paths': ['r'], 'batch_size': 10}`.
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
            iterable_of_texts = [d.text for d in document_batch]
            embedding_matrix = self.tfidf_vectorizer.transform(iterable_of_texts)
            for doc, doc_embedding in zip(document_batch, embedding_matrix):
                doc.embedding = doc_embedding
