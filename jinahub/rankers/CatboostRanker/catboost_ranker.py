__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
from typing import Dict, List, Optional

import numpy as np
from catboost import CatBoostRanker, Pool

from jina import DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger


class CatboostRanker(Executor):
    def __init__(
        self,
        query_features: List[str],
        match_features: List[str],
        label: str,
        weight: Optional[str] = None,
        model_path: Optional[str] = None,
        catboost_parameters: Dict = {
            'iterations': 2000,
            'custom_metric': ['NDCG', 'AverageGain:top=10'],
            'verbose': False,
            'random_seed': 0,
            'loss_function': 'QuerySoftMax',
        },
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.q_features = query_features
        self.m_features = match_features
        self.label = label
        self.weight = weight
        self.model_path = model_path
        self.catboost_parameters = catboost_parameters
        self.model = None
        self.logger = JinaLogger('catboost logger')
        if self.model_path and os.path.exists(self.model_path):
            self._load_model()
        else:
            self.model = CatBoostRanker()

    def _load_model(self, path):
        """Load model from model path"""
        self.model = CatBoostRanker().load_model(path)

    def _save_model(self, path):
        """Dump model into cbm."""
        self.model.save_model(fname=path, format='cbm')

    def _extract_features(self, docs: DocumentArray):
        """Loop over docs, build dataset to train the model.

        This func get all :attr:`query_feature` from tags inside :class:`Document`,
        and get all :attr:`match_features` from all matched :class:`Document` for each query document.
        Each query document corresponded to N matched documents, and each query-document pair features
        are combined into a feature vector. We stack all feature vectors and return the training data.
        """
        group_ids = []
        label_vector = []
        feature_vectors = []
        for idx, doc in enumerate(docs):
            q_feature_vector = [
                doc.tags.get(query_feature) for query_feature in self.q_features
            ]
            for match in doc.matches:
                group_ids.append(idx)
                label_vector.append(match.tags.get(self.label))
                m_feature_vector = [
                    match.tags.get(docum_feature) for docum_feature in self.m_features
                ]
                feature_vectors.append(q_feature_vector + m_feature_vector)
        return np.array(feature_vectors), label_vector, group_ids

    def _extract_weights(self, docs: DocumentArray):
        """Extract weights data from docs, weights are needed if the query_docs has different
        importance. Once given, we'll get values stored in the Document.tags based on the field name.
        """
        weight_vector = []
        for doc in docs:
            for _ in doc.matches:
                # weight need to have the same size of matches, while get from doc.
                weight_vector.append(doc.tags.get(self.weight))
        return weight_vector

    def build_catboost_pool(self, docs: DocumentArray):
        """Create the catboost :class:`Pool` for training.

        :param docs: A :class:`DocumentArray` to build the pool.
        :return: A :class:`Pool` object of catboost.
        """
        data, label, group_id = self._extract_features(docs)
        if self.weight:
            return Pool(
                data=data,
                label=label,
                group_id=group_id,
                weight=self._extract_weights(docs),
            )
        else:
            return Pool(data=data, label=label, group_id=group_id)

    @requests(on='/train')
    def train(self, docs: DocumentArray, parameters: Dict = {}, **kwargs):
        """This endpoint fits a catboost ranker based on docs.

        :param docs: Documents to fit the model, each docs should have fields corresponded to
          the :attr:`query_features` and has matches with corresponded :attr:`match_features` to
          extract feature from. Also, a label field should be stored in each match document.
        :param parameters: Parameters to set, we'll get :attr:`catboost_parameters` from the parameters field.
        """
        catboost_parameters = parameters.get(
            'catboost_parameters', self.catboost_parameters
        )
        train_pool = self.build_catboost_pool(docs)
        self.model = CatBoostRanker(**catboost_parameters)
        self.model.fit(train_pool)

    @requests(on='/rank')
    def rank(self, docs: DocumentArray, parameters: Dict = {}, **kwargs):
        """This endpoint ranks a catboost ranker based on docs.

        :param docs: Documents to fit the model, each docs should have fields corresponded to
          the :attr:`query_features` and has matches with corresponded :attr:`match_features` to
          extract feature from. An additional field will be attached to the matched documents as was defined
          as :attr:`label`.
        :param parameters: Parameters to set, we'll get :attr:`catboost_parameters` from the parameters field.
        """
        feature_vecs, _, _ = self._extract_features(docs)
        if not self.model.is_fitted():
            raise ValueError(
                'You need to train your model before make prediction, call `train` endpoint first.'
            )
        preds = self.model.predict(feature_vecs)
        matches = docs.traverse_flat(traversal_paths=['m'])
        for match, pred in zip(matches, preds):
            match.tags[self.label] = pred

    @requests(on='/dump')
    def dump(self, parameters: Dict, **kwargs):
        """Dump trained model to specified path

        :param parameters: Parameters pass to this endpoint, expect :attr:`model_path` to be set.
        """
        model_path = parameters.get('model_path', None)
        if model_path:
            self._save_model(model_path)
        else:
            raise ValueError('Please specify a `model_path` inside parameters variable')

    @requests(on='/load')
    def load(self, parameters: Dict, **kwargs):
        """Load trained model from specified path

        :param parameters: Parameters pass to this endpoint, expect :attr:`model_path` to be set.
        """
        model_path = parameters.get('model_path', self.model_path)
        if model_path:
            self._load_model(model_path)
        else:
            raise FileNotFoundError(
                f'Model {model_path} does not exist. Please specify the correct model_path inside parameters.'
            )
