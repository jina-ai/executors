__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import copy
from typing import Dict

from jina import requests, DocumentArray, Executor

from jina_commons import get_logger
from jinahub.indexers.searcher.NumpySearcher import NumpySearcher
from jinahub.indexers.storage.PostgreSQLStorage import PostgreSQLStorage


class NumpyPostgresSearcher(Executor):
    """A Compound Indexer made up of a NumpyIndexer (for vectors) and a Postgres Indexer"""

    def __init__(
        self,
        dump_path=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # when constructed from rolling update the dump_path is passed via a runtime_arg
        dump_path = dump_path or kwargs.get('runtime_args').get('dump_path')
        self.logger = get_logger(self)
        self._kv_indexer = None
        self._vec_indexer = None
        if dump_path:
            self._vec_indexer = NumpySearcher(dump_path=dump_path, **kwargs)
            self._kv_indexer = PostgreSQLStorage(**kwargs)
        else:
            self.logger.warning(
                f'No dump path provided for {self}. Use .rolling_update() to re-initialize...'
            )

    @requests(on='/search')
    def search(self, docs: 'DocumentArray', parameters: Dict = None, **kwargs):
        if self._kv_indexer and self._vec_indexer:
            self._vec_indexer.search(docs, parameters)
            kv_parameters = copy.deepcopy(parameters)
            kv_parameters['traversal_paths'] = [
                path + 'm' for path in kv_parameters.get('traversal_paths', ['r'])
            ]
            self._kv_indexer.search(docs, kv_parameters)
        else:
            return
