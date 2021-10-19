import os
import random
import time
from typing import Dict, OrderedDict

import numpy as np
import pytest
from jina import Document, DocumentArray, Executor, Flow, requests
from jina_commons.indexers.dump import dump_docs

from jinahub.indexers.compound.FaissLMDBSearcher.faiss_lmdb import FaissLMDBSearcher
from jinahub.indexers.storage.LMDBStorage.lmdb_storage import LMDBStorage

random.seed(0)
np.random.seed(0)

cur_dir = os.path.dirname(os.path.abspath(__file__))
ORIGIN_TAG = 'origin'
TOP_K = 100


class TagMatchMerger(Executor):
    @requests(on='/tag_search')
    def merge(self, docs_matrix, parameters: Dict, **kwargs):
        if docs_matrix:
            # noinspection PyTypeHints
            results = OrderedDict()
            for docs in docs_matrix:
                for doc in docs:
                    if doc.id in results:
                        results[doc.id].matches.extend(doc.matches)
                    else:
                        results[doc.id] = doc

            top_k = parameters.get('top_k')
            if top_k:
                top_k = int(top_k)

            for doc in results.values():
                doc.matches = sorted(
                    doc.matches,
                    key=lambda m: m.scores['euclidean'].value,
                    reverse=True,
                )[:top_k]

            docs = DocumentArray(list(results.values()))
            return docs


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


class FaissTaggingFileSearcher(FaissLMDBSearcher):
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
        d.embedding = np.random.random([embed_dim]).astype(dtype=np.float32)
        yield d


def validate_diff_sources(results, num_shards, docs_before: DocumentArray):
    distinct_shards = {}
    for doc in results[0].docs:
        for match in doc.matches:
            if match.tags[ORIGIN_TAG] not in distinct_shards:
                distinct_shards[match.tags[ORIGIN_TAG]] = 0
            distinct_shards[match.tags[ORIGIN_TAG]] += 1
    # TODO: distinct_shards is not determined
    # np.testing.assert_equal(len(distinct_shards.keys()), num_shards)
    np.testing.assert_equal(sum(distinct_shards.values()), TOP_K)


# TODO we do not support shards=1 for replicas>1
def assert_folder(dump_path, num_shards):
    assert os.path.exists(dump_path)
    for i in range(num_shards):
        assert os.path.exists(os.path.join(dump_path, str(i)))
        assert os.path.exists(os.path.join(dump_path, str(i), 'ids'))
        assert os.path.exists(os.path.join(dump_path, str(i), 'vectors'))
        assert os.path.exists(os.path.join(dump_path, str(i), 'metas'))


@pytest.mark.parametrize('num_shards', (2, 3, 7))
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
