__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
from typing import Dict, Optional, List

import lightgbm
import numpy as np
from jina.logging.logger import JinaLogger
from jina import Executor, DocumentArray, requests


class LightGBMRanker(Executor):
    """
    Computes a relevance score for each match using a pretrained Ltr model trained with LightGBM (https://lightgbm.readthedocs.io/en/latest/index.html)
    :param query_features: name of the features to extract from query Documents and used to compute relevance scores by the model loaded
    from model_path
    :param match_features: name of the features to extract from match Documents and used to compute relevance scores by the model loaded
    from model_path
    :param relevance_label: If call :meth:`train` endpoint, the label will be used as groundtruth for model training. If on :meth:`rank` endpoint, the
    label will be used to assign a score to :attr:`Document.scores` field.
    :param model_path: path to the pretrained model previously trained using LightGBM.
    :param params: Parameters used to train the LightGBM learning-to-rank model.
    :param categorical_query_features: name of features contained in `query_features` corresponding to categorical features.
    :param categorical_match_features: name of features contained in `match_features` corresponding to categorical features.
    :param query_features_before: True if `query_feature_names` must be placed before the `match` ones in the `dataset` used for prediction.
    :param args: Additional positional arguments
    :param kwargs: Additional keyword arguments
    .. note::
        The name of the features are used to extract the features from incoming `documents`. Check how these features are accessed in
        :class:`Document` at https://docs.jina.ai/api/jina.types.document/
    """

    def __init__(
        self,
        query_features: List[str],
        match_features: List[str],
        relevance_label: str,
        model_path: Optional[str] = None,
        params: Dict = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'lambdarank',
            'min_data_in_leaf': 1,
            'feature_pre_filter': False,
        },
        categorical_query_features: Optional[List[str]] = None,
        categorical_match_features: Optional[List[str]] = None,
        query_features_before: bool = False,
        *args,
        **kwargs,
    ):
        super(LightGBMRanker, self).__init__(*args, **kwargs)
        self.params = params
        self.model_path = model_path
        self.query_features = query_features
        self.match_features = match_features
        self.categorical_query_features = categorical_query_features
        self.categorical_match_features = categorical_match_features
        self.query_features_before = query_features_before
        self.label = relevance_label
        self.logger = JinaLogger(self.__class__.__name__)
        if self.model_path and os.path.exists(self.model_path):
            self.booster = self._load_model(self.model_path)
        else:
            self.booster = None

    def _load_model(self, path):
        """Load model from model path"""
        self.booster = lightgbm.Booster(model_file=path)
        model_num_features = self.booster.num_feature()
        expected_num_features = len(self.query_features + self.match_features)
        if self.categorical_query_features:
            expected_num_features += len(self.categorical_query_features)
        if self.categorical_match_features:
            expected_num_features += len(self.categorical_match_features)
        if model_num_features != expected_num_features:
            raise ValueError(
                f'The number of features expected by the LightGBM model {model_num_features} is different'
                f'than the ones provided in input {expected_num_features}'
            )

    def _save_model(self, path):
        """Dump model into cbm."""
        self.booster.save_model(model_file=path)

    def _get_features_dataset(self, docs: DocumentArray) -> 'lightgbm.Dataset':
        q_features, m_features, group, labels = [], [], [], []
        query_feature_names = self.query_features
        if self.categorical_query_features:
            query_feature_names += self.categorical_query_features
        match_feature_names = self.match_features
        if self.categorical_match_features:
            match_feature_names += self.categorical_match_features
        for doc in docs:
            query_feature = []
            match_feature = []
            query_values = [doc.tags.get(feature) for feature in query_feature_names]
            for match in doc.matches:
                match_values = [
                    match.tags.get(feature) for feature in match_feature_names
                ]
                labels.append(match.tags.get(self.label))
                match_feature.append(match_values)
                query_feature.append(query_values)
            group.append(len(doc.matches))
            q_features.append(query_feature)
            m_features.append(match_feature)

        query_dataset = lightgbm.Dataset(
            data=np.vstack(q_features),
            group=group,
            feature_name=query_feature_names,
            categorical_feature=self.categorical_query_features,
            free_raw_data=False,
        )

        match_dataset = lightgbm.Dataset(
            data=np.vstack(m_features),
            group=group,
            label=labels,
            feature_name=match_feature_names,
            categorical_feature=self.categorical_match_features,
            free_raw_data=False,
        )
        if self.query_features_before:
            return query_dataset.construct().add_features_from(
                match_dataset.construct()
            )
        else:
            return match_dataset.construct().add_features_from(
                query_dataset.construct()
            )

    @requests(on='/train')
    def train(self, docs: DocumentArray, **kwargs):
        """The :meth:`train` endpoint allows user to train the lightgbm ranker
        in an incremental manner. The features will be extracted from the `attr`:`tags`,
        including all the :attr:`query_features` and :attr:`match_features`. The label/groundtruth of the
        training data will be the :attr:`label` field.

        :param docs: :class:`DocumentArray` passed by the user or previous executor.
        :param kwargs: Additional key value arguments.
        """
        train_set = self._get_features_dataset(docs)
        categorical_feature = []
        if self.categorical_query_features:
            categorical_feature += self.categorical_query_features
        if self.categorical_match_features:
            categorical_feature += self.categorical_match_features
        if not categorical_feature:
            categorical_feature = 'auto'
        self.booster = lightgbm.train(
            train_set=train_set,
            init_model=self.booster,
            params=self.params,
            keep_training_booster=True,
            categorical_feature=categorical_feature,
        )

    @requests(on='/search')
    def rank(self, docs: DocumentArray, **kwargs):
        """The :meth:`rank` endpoint allows user to assign a score to their docs given by pre-trained
          :class:`LightGBMRanker`. Once load, the :class:`LightGBMRanker` will load the pre-trained model
          and make predictions on the documents. The predictions are made based on extracted dataset from
          query and matches. The :attr:`query_features` will be extracted from query document :attr:`tags`
          and `match_features` will be extracted from corresponded matches documents tags w.r.t the query document.

        :param docs: :class:`DocumentArray` passed by the user or previous executor.
        :param kwargs: Additional key value arguments.
        """
        dataset = self._get_features_dataset(docs)
        predictions = self.booster.predict(dataset.get_data())
        matches = docs.traverse_flat(traversal_paths=['m'])
        for prediction, match in zip(predictions, matches):
            match.scores[self.label] = prediction
        for doc in docs:
            doc.matches.sort(key=lambda x: x.scores[self.label].value, reverse=True)

    @requests(on='/dump')
    def dump(self, parameters: Dict, **kwargs):
        """Dump trained model to specified path

        :param parameters: Parameters pass to this endpoint, expect :attr:`model_path` to be set.
        """
        model_path = parameters.get('model_path', None)
        if model_path:
            self._save_model(model_path)
        else:
            raise ValueError(
                'Please specify the `model_path` inside parameters variable'
            )

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
                f'Model {model_path} does not exist. Please specify the model_path inside parameters.'
            )
