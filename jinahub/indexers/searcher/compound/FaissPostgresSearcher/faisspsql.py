__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import copy
from typing import Dict

from jina import requests, DocumentArray, Executor

from jina_commons import get_logger

try:
    from jinahub.indexers.searcher.FaissSearcher import FaissSearcher
except:
    from jina_executors.indexers.searcher.FaissSearcher.faiss_searcher import (
        FaissSearcher,
    )

try:
    from jinahub.indexers.storage.PostgreSQLStorage import (
        PostgreSQLStorage,
    )
except:
    from jina_executors.indexers.storage.PostgreSQLStorage import (
        PostgreSQLStorage,
    )


class FaissPostgresSearcher(Executor):
    """A Compound Indexer made up of a FaissSearcher (for vectors) and a Postgres Indexer"""

    def __init__(
        self,
        dump_path=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # when constructed from rolling update the dump_path is passed via a runtime_arg
        self._init_kwargs = kwargs
        dump_path = dump_path or self._init_kwargs.get('runtime_args').get('dump_path')
        self.logger = get_logger(self)
        self._kv_indexer = PostgreSQLStorage(**self._init_kwargs)
        if dump_path:
            self._vec_indexer = FaissSearcher(dump_path=dump_path, **self._init_kwargs)
        else:
            self._vec_indexer = None
            self.logger.warning(
                f'No dump path provided for {self}. Use /reload to re-initialize...'
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
            self.logger.warning(
                'Not all sub-indexers initialized. Use /reload to re-initialize...'
            )

    @requests(on='/dump')
    def dump(self, parameters: Dict, **kwargs):
        """Dump the index

        :param parameters: a dictionary containing the parameters for the dump
        """
        self._kv_indexer.dump(parameters, **kwargs)

    @requests(on='/reload')
    def reload(self, parameters: Dict, **kwargs):
        dump_path = parameters.get('dump_path', None)
        if dump_path is None:
            self.logger.error(f'No "dump_path" provided for {self}')
            return

        self._vec_indexer = FaissSearcher(dump_path=dump_path, **self._init_kwargs)

    @requests(on='/index')
    def index(self, **kwargs):
        self._kv_indexer.add(**kwargs)

    @requests(on='/update')
    def update(self, **kwargs):
        self._kv_indexer.update(**kwargs)

    @requests(on='/delete')
    def delete(self, **kwargs):
        self._kv_indexer.delete(**kwargs)
