__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from itertools import groupby
from typing import List, Dict

from jina import Executor, requests, DocumentArray
from jina.logging.logger import JinaLogger


class MinRanker(Executor):
    """
    :class:`MinRanker` aggregates the score
    of the matched doc from the matched chunks.
    For each matched doc, the score is aggregated
    from all the matched chunks belonging to that doc.
    :param metric: the distance metric used in `scores`
    :param default_traversal_paths: traverse path on docs, e.g. ['r'], ['c']
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
    """

    def __init__(self,
                 metric: str = None,
                 default_traversal_paths: List[str] = None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = JinaLogger(self.__class__.__name__)
        if not metric:
            self.logger.error('metric should not be None')
        self.metric = metric
        self.default_traversal_paths = default_traversal_paths or ['r']


    @requests(on='/search')
    def rank(self, docs: DocumentArray, parameters: Dict, *args, **kwargs):
        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )

        for doc in docs.traverse_flat(traversal_paths):
            matches_of_chunks = []
            for chunk in doc.chunks:
                matches_of_chunks.extend(chunk.matches)
            groups = groupby(sorted(matches_of_chunks, key=lambda d: d.parent_id), lambda d: d.parent_id)
            for key, group in groups:
                chunk_match_list = list(group)
                chunk_match_list.sort(key=lambda m: -m.scores[self.metric].value)
                match = chunk_match_list[0]
                match.id = chunk_match_list[0].parent_id
                doc.matches.append(match)
            doc.matches.sort(key=lambda d: -d.scores[self.metric].value)
