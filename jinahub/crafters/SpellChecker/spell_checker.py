__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
import pickle
from typing import Dict, Iterable

from jina import DocumentArray, Executor, requests
from jina.excepts import PretrainedModelFileDoesNotExist

cur_dir = os.path.dirname(os.path.abspath(__file__))


class SpellChecker(Executor):
    def __init__(
        self,
        model_path: str = os.path.join(cur_dir, 'model/model.pickle'),
        traversal_paths: Iterable = ['r'],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_path = model_path
        self.traversal_paths = traversal_paths

        self.model = None
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as model_file:
                self.model = pickle.load(model_file)
        else:
            raise PretrainedModelFileDoesNotExist(
                f'{self.model_path} not found, cannot find a fitted spell checker'
            )

    @requests
    def spell_check(self, docs: DocumentArray, parameters: Dict = {}, **kwargs):
        for d in docs.traverse_flat(
            parameters.get('traversal_paths', self.traversal_paths)
        ):
            d.content = ' '.join(self.model.transform(d.content))
