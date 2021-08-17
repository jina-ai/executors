__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from collections import OrderedDict
from typing import List, Tuple

from jina import Executor, requests, DocumentArray


class SimpleMerger(Executor):
    """
    The SimpleMerger merges the results of shards by appending all matches..

    :param default_traversal_paths: traverse path on docs, e.g. ['r'], ['c']
    :param args: additional arguments
    :param kwargs: additional key value arguments
    """

    def __init__(self, default_traversal_paths: Tuple[str] = ('r',), **kwargs):

        self.default_traversal_paths = default_traversal_paths
        super().__init__(**kwargs)

    @requests
    def merge(self, docs_matrix, parameters: dict, **kwargs):
        traversal_paths = parameters.get('traversal_paths', self.default_traversal_paths)
        results = {}
        for docs in docs_matrix:
            for doc in docs.traverse_flat(traversal_paths):
                if doc.id in results:
                    results[doc.id].matches.extend(doc.matches)
                else:
                    results[doc.id] = doc
        return DocumentArray(list(results.values()))
