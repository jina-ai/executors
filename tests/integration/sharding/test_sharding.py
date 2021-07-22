import os
import random
import time
from typing import Dict

import numpy as np
import pytest
from jina import Document, Flow, DocumentArray, requests

from jina_commons.indexers.dump import dump_docs
from jinahub.indexers.searcher.compound.NumpyLMDBSearcher import NumpyLMDBSearcher
from jinahub.indexers.storage.LMDBStorage import LMDBStorage
from tests.integration.psql_dump_reload.test_dump_psql import (
    MatchMerger,
)

random.seed(0)
np.random.seed(0)

cur_dir = os.path.dirname(os.path.abspath(__file__))
ORIGIN_TAG = 'origin'
TOP_K = 30


class TagMatchMerger(MatchMerger):
    @requests(on='/tag_search')
    def merge(self, docs_matrix, parameters: Dict, **kwargs):
        MatchMerger.merge(
            self, docs_matrix=docs_matrix, parameters=parameters, **kwargs
        )


class TaggingFileSearcher(LMDBStorage):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def search(self, docs: DocumentArray, parameters: Dict = None, **kwargs) -> None:
        # TODO shouldn't be necessary
        parameters = {'traversal_paths': ['m']}
        LMDBStorage.search(self, docs, parameters=parameters, **kwargs)
        for doc in docs:
            for match in doc.matches:
                match.tags[ORIGIN_TAG] = self.runtime_args.pea_id


class NumpyTaggingFileSearcher(NumpyLMDBSearcher):
    def __init__(
        self,
        dump_path=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._kv_indexer = TaggingFileSearcher(dump_path=dump_path, **kwargs)

    @requests(on='/tag_search')
    def search(self, docs: 'DocumentArray', parameters: Dict = None, **kwargs):
        super().search(docs, parameters, **kwargs)


def random_docs(start, end, embed_dim=10):
    for j in range(start, end):
        d = Document()
        d.content = f'hello world from {j}'
        d.embedding = np.random.random([embed_dim])
        yield d


def validate_diff_sources(results, num_shards, docs_before: DocumentArray):
    distinct_shards = {}
    for doc in results[0].docs:
        for match in doc.matches:
            if match.tags[ORIGIN_TAG] not in distinct_shards:
                distinct_shards[match.tags[ORIGIN_TAG]] = 0
            distinct_shards[match.tags[ORIGIN_TAG]] += 1
    np.testing.assert_equal(len(distinct_shards.keys()), num_shards)
    np.testing.assert_equal(sum(distinct_shards.values()), TOP_K)


# TODO we do not support shards=1 for replicas>1
def assert_folder(dump_path, num_shards):
    assert os.path.exists(dump_path)
    for i in range(num_shards):
        assert os.path.exists(os.path.join(dump_path, str(i)))
        assert os.path.exists(os.path.join(dump_path, str(i), 'ids'))
        assert os.path.exists(os.path.join(dump_path, str(i), 'vectors'))
        assert os.path.exists(os.path.join(dump_path, str(i), 'metas'))


# TODO: add num_shards=7
@pytest.mark.parametrize('num_shards', (2, 3))
def test_shards_numpy_filequery(tmpdir, num_shards):
    pod_name = 'index'
    os.environ['WORKSPACE'] = str(tmpdir)
    os.environ['SHARDS'] = str(num_shards)

    docs_indexed = list(random_docs(0, 201))
    dump_path = os.path.join(tmpdir, 'dump_path')
    dump_docs(docs_indexed, dump_path, num_shards)

    assert_folder(dump_path, num_shards)

    inputs = list(random_docs(0, 1))

    # TODO workspace is wrongly saved to curdir
    with Flow.load_config('flow.yml') as flow:
        flow.rolling_update(pod_name=pod_name, dump_path=dump_path)
        time.sleep(2)
        results = flow.post(
            on='/tag_search',
            inputs=inputs,
            parameters={'top_k': TOP_K},
            return_results=True,
        )
        validate_diff_sources(results, num_shards, docs_indexed)
