__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from itertools import groupby
from typing import Iterable, Dict

from jina import Executor, requests, DocumentArray


class MinRanker(Executor):
    """
    :class:`MinRanker` aggregates the score of the matched doc from the matched chunks.
    For each matched doc, the score is aggregated from all the matched chunks belonging to that doc.

    :param metric: the distance metric used in `scores`
    :param default_traversal_paths: traverse path on docs, e.g. ['r'], ['c']
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
    """

    def __init__(
        self,
        metric: str,
        default_traversal_paths: Iterable[str] = ('r',),
        metric_is_distance: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.metric = metric
        if metric_is_distance:
            self.distance_mult = 1
        else:
            self.distance_mult = -1
        self.default_traversal_paths = default_traversal_paths

    @requests(on='/search')
    def rank(self, docs: DocumentArray, parameters: Dict, *args, **kwargs):
        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )

        for doc in docs.traverse_flat(traversal_paths):
            matches_of_chunks = []
            for chunk in doc.chunks:
                matches_of_chunks.extend(chunk.matches)
            groups = groupby(
                sorted(matches_of_chunks, key=lambda d: d.parent_id),
                lambda d: d.parent_id,
            )
            for key, group in groups:
                chunk_match_list = list(group)
                chunk_match_list.sort(key=lambda m: self.distance_mult * m.scores[self.metric].value)
                match = chunk_match_list[0]
                match.id = chunk_match_list[0].parent_id
                doc.matches.append(match)
            doc.matches.sort(key=lambda d: self.distance_mult * d.scores[self.metric].value)
