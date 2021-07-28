import numpy as np
from catboost import CatBoostRanker, Pool

from jina import DocumentArray, Executor, requests
from jina_commons.batching import get_docs_batch_generator


class CatBoostRanker(Executor):
    def __init__(
        self,
        query_features: List[str],
        document_features: List[str],
        label: str,
        model_path: str = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.q_features = query_features
        self.d_features = document_features
        self.label = label
        self.model_path = model_path

    def _extract_features_from_docs(self, docs: DocumentArray):
        """Loop over docs, build dataset to train the model.

        This func get all :attr:`query_feature` from tags inside :class:`Document`,
        and get all :attr:`document_features` from all matched :class:`Document` for each query document.
        Each query document corresponded to N matched documents, and each query-document pair features
        are combined into a feature vector. We stack all feature vectors and return the training data.
        """
        group_ids = []
        feature_vectors = []
        for idx, doc in enumerate(docs):
            group_ids.append(idx)
            q_feature_vector = [
                doc.tags.get(query_feature) for query_feature in self.q_features
            ]
            for match in doc.matches:
                d_feature_vector = [
                    match.tags.get(docum_feature) for docum_feature in self.d_features
                ]
                feature_vectors.append(q_feature_vector + d_feature_vector)
        feature_vectors = np.array(feature_vectors)
        return Pool(data=feature_vectors, label=self.label, group_id=group_ids)
