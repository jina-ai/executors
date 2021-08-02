from typing import Dict

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
        catboost_parameters: Dict = {
            'iterations': 2000,
            'custom_metric': ['NDCG', 'PFound', 'AverageGain:top=10'],
            'verbose': False,
            'random_seed': 0,
            'loss_function': 'RMSE',
        },
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.q_features = query_features
        self.d_features = document_features
        self.label = label
        self.model_path = model_path
        self.catboost_parameters = catboost_parameters
        self.model = None
        if self.model_path and os.path.exists(self.model_path):
            self.model = self._load_model()

    def _load_model(self):
        """Load model from model path"""
        self.model = CatBoostRanker().load_model(self.model_path)

    def _save_model(self):
        """Dump model into cbm."""
        CatBoostRanker.save_model(self.model_path, format='cbm')

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

    @requests(on='train')
    def train(self, parameters: Dict, **kwargs):
        catboost_parameters = parameters.get(
            'catboost_parameters', self.catboost_parameters
        )
        self.model = CatBoostRanker(**catboost_parameters)
        self.model.fit(train_pool)

    @requests(on='/dump')
    def dump(self, parameters: Dict, **kwargs):
        model_path = parameters.get('model_path', None)
        if model_path:
            self._save_model(model_path)
        else:
            self._save_model('tmp/model')
            logger.info(
                f'Model {model_path} does not exist. Model has been saved to tmp/model.cbm.'
            )

    @requests(on='/load')
    def load(self, parameters: Dict, **kwargs):
        model_path = parameters.get('model_path', self.model_path)
        if model_path:
            self._load_model(model_path)
        else:
            logger.warning(
                f'Model {model_path} does not exist. Please specify the correct model_path inside parameters.'
            )
