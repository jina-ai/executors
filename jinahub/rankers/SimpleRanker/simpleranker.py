__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from collections import defaultdict
from typing import Dict, Iterable

from jina import Document, DocumentArray, Executor, requests


class SimpleRanker(Executor):
    """
    SimpleRanker aggregates the score of matched chunks (matches that are chunks) to the
    score of the parent document of the matched chunks. The Document's matches are then
    replaced by matches based on the parent documents of the chunks - they contain an
    `id` and the aggregated score only.

    This ranker is used to "bubble-up" the scores of matched chunks to the scores
    of the parent document. 
    """

    def __init__(
        self,
        metric: str = 'cosine',
        ranking: str = 'min',
        traversal_paths: Iterable[str] = ('r',),
        matches_path: str = 'cm',
        *args,
        **kwargs,
    ):
        """
        :param metric: the distance metric used in `scores`
        :param ranking: The sort and aggregation function that the executor uses.
            The allowed options are:
            - `min`: Set the chunk's score to the minimum score of its matches, sort
                chunks in an ascending order.
            - `max`: Set the match's score to the maximum score of its matches, sort
                chunks in a descending order.
            - `mean_min`: Set the match's score to the mean score of its matches, sort
                chunks in an ascending order.
            - `mean_max`: Set the match's score to the mean score of its matches, sort
                chunks in an decending order.
        :param traversal_paths: The traversal paths, used to obtain the documents we
            want the ranker to work on - these are the "query" documents, for which
            we wish to create aggregated matches.
        :param matches_path: The path (starting from the "query" document) where the
            matched chunks can be found. The default here is "cm", which means that
            we expecte the matched chunks to be the matches of the chunks of our query
            document. This path should end with "m"
        """
        super().__init__(*args, **kwargs)

        if ranking not in ['min', 'max', 'mean_min', 'mean_max']:
            raise ValueError(
                'ranking should be one of "min", "max", "mean_min", "mean_max"',
                f' got "{ranking}"',
            )

        if matches_path[-1] != "m":
            raise ValueError(f'matches_path should end with "m", got {matches_path}')

        self.metric = metric
        self.ranking = ranking
        self.traversal_paths = traversal_paths
        self.matches_path = matches_path

    @requests(on='/search')
    def rank(self, docs: Optional[DocumentArray] = None, parameters: Dict = {}, **kwargs):
        """Aggregate the score of matched chunks to the score of their parent document.

        The matches of query documents that are passed in `docs` are replaced (if they
        exist) by documents based on the aggregated score of the parent documents of
        matched chunks - that is, by documents containing only parent id of matched
        chunks, and the aggregated score corresponding to that id.

        :param docs: The documents for which to create aggregated matches (specifically,
            the aggregated matches will be created for documents that are on the
            traversal paths of documents passed in this argument).
        :param parameters: Extra parameters that can be used to override the parameters
            set at creation of the Executor. Valid values are `traversal_paths`
        """
        if docs is None:
            return

        traversal_paths = parameters.get('traversal_paths', self.traversal_paths)
        for doc in docs.traverse_flat(traversal_paths):
            parents_scores = defaultdict(list)
            for m in DocumentArray([doc]).traverse_flat([self.matches_path]):
                parents_scores[m.parent_id].append(m.scores[self.metric].value)

            # Aggregate match scores for parent document and
            # create doc's match based on parent document of matched chunks
            new_matches = []
            for match_parent_id, scores in parents_scores.items():
                if self.ranking == 'min':
                    score = min(scores)
                elif self.ranking == 'max':
                    score = max(scores)
                elif self.ranking in ['mean_min', 'mean_max']:
                    score = sum(scores) / len(scores)

                new_matches.append(
                    Document(id=match_parent_id, scores={self.metric: score})
                )

            # Sort the matches
            doc.matches = new_matches
            if self.ranking in ['min', 'mean_min']:
                doc.matches.sort(key=lambda d: d.scores[self.metric].value)
            elif self.ranking in ['max', 'mean_max']:
                doc.matches.sort(key=lambda d: -d.scores[self.metric].value)
