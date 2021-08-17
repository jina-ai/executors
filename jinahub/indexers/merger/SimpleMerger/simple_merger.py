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

    def __init__(self, default_traversal_paths: Tuple[str] = ('r',), is_unique: bool = False, **kwargs):
        self.is_unique = is_unique
        self.default_traversal_paths = default_traversal_paths
        super().__init__(**kwargs)

    @requests
    def merge(self, docs_matrix: List[DocumentArray], parameters: dict, **kwargs):
        traversal_paths = parameters.get('traversal_paths', self.default_traversal_paths)
        is_unique = parameters.get('is_unique', self.is_unique)
        results = {}
        for docs in docs_matrix:
            for doc in docs.traverse_flat(traversal_paths):
                if doc.id in results:
                    if is_unique:
                        for match in doc.matches:
                            if match not in results[doc.id].matches:
                                results[doc.id].matches.add(match)
                    else:
                        results[doc.id].matches.extend(doc.matches)
                else:
                    results[doc.id] = doc
        return DocumentArray(list(results.values()))
