__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
import pickle
from typing import Dict, Iterable

from jina import DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger
from pyngramspell import PyNgramSpell

cur_dir = os.path.dirname(os.path.abspath(__file__))


class SpellChecker(Executor):
    def __init__(
        self,
        model_path: str = os.path.join(cur_dir, 'model.pickle'),
        traversal_paths: Iterable = ['r'],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.traversal_paths = traversal_paths
        self.logger = JinaLogger(self.metas.name)

        self.model_path = model_path

        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as model_file:
                self.model = pickle.load(model_file)
        else:
            self.logger.warning(f'model_path {self.model_path} is empty. Use /train')

    @requests(on='/train')
    def train(self, docs: DocumentArray, parameters: Dict = {}, **kwargs):
        speller = PyNgramSpell(min_freq=0)
        input_training_data = [d.content for d in docs]
        speller.fit(input_training_data)
        speller.save(self.model_path)

    @requests(on=['/index', '/search', '/update', '/delete'])
    def spell_check(self, docs: DocumentArray, parameters: Dict = {}, **kwargs):
        for d in docs.traverse_flat(
            parameters.get('traversal_paths', self.traversal_paths)
        ):
            d.content = self.model.transform(d.content)
