import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
from jina import Flow, Document, Executor, DocumentArray, requests
from jina.logging.profile import TimeContext
from jina_commons.indexers.dump import (
    import_vectors,
    import_metas,
)

from jinahub.indexers.storage.LMDBStorage import LMDBStorage
from jinahub.indexers.storage.PostgreSQLStorage.postgreshandler import (
    doc_without_embedding,
)

# REQUIRED INDEXERS

# REQUIRED INDEXERS

cur_dir = os.path.dirname(os.path.abspath(__file__))
storage_flow_yml = os.path.join(cur_dir, 'flow_storage.yml')
query_flow_yml = os.path.join(cur_dir, 'flow_query.yml')


@pytest.fixture()
def docker_compose(request):
    os.system(
        f"docker-compose -f {request.param} --project-directory . up  --build -d --remove-orphans"
    )
    time.sleep(5)
    yield
    os.system(
        f"docker-compose -f {request.param} --project-directory . down --remove-orphans"
    )


class Pass(Executor):
    @requests(on='/search')
    def pass_me(self, **kwargs):
        pass


class MatchMerger(Executor):
    @requests(on='/search')
    def merge(self, docs_matrix, parameters: Dict, **kwargs):
        if docs_matrix:
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
                    key=lambda m: m.scores['similarity'].value,
                    reverse=True,
                )[:top_k]

            docs = DocumentArray(list(results.values()))
            return docs


def get_documents(nr=10, index_start=0, emb_size=7):
    for i in range(index_start, nr + index_start):
        with Document(id=i) as d:
            d.text = f'hello world {i}'
            d.embedding = np.random.random(emb_size)
            d.tags['field'] = f'tag data {i}'
        yield d


def assert_dump_data(dump_path, docs, shards, pea_id):
    size_shard = len(docs) // shards
    size_shard_modulus = len(docs) % shards
    ids_dump, vectors_dump = import_vectors(
        dump_path,
        str(pea_id),
    )
    if pea_id == shards - 1:
        docs_expected = docs[
            (pea_id) * size_shard : (pea_id + 1) * size_shard + size_shard_modulus
        ]
    else:
        docs_expected = docs[(pea_id) * size_shard : (pea_id + 1) * size_shard]
    print(f'### pea {pea_id} has {len(docs_expected)} docs')

    # TODO these might fail if we implement any ordering of elements on dumping / reloading
    ids_dump = list(ids_dump)
    vectors_dump = list(vectors_dump)
    np.testing.assert_equal(ids_dump, [d.id for d in docs_expected])
    np.testing.assert_allclose(vectors_dump, [d.embedding for d in docs_expected])

    _, metas_dump = import_metas(
        dump_path,
        str(pea_id),
    )
    metas_dump = list(metas_dump)
    np.testing.assert_equal(
        metas_dump,
        [doc_without_embedding(d) for d in docs_expected],
    )


def path_size(dump_path):
    dir_size = (
        sum(f.stat().st_size for f in Path(dump_path).glob('**/*') if f.is_file()) / 1e6
    )
    return dir_size


# replicas w 1 shard doesn't work
# @pytest.mark.parametrize('shards', [1, 3, 7])
@pytest.mark.parametrize('shards', [3, 7])
@pytest.mark.parametrize('nr_docs', [10])
@pytest.mark.parametrize('emb_size', [10])
def test_dump_reload(tmpdir, nr_docs, emb_size, shards):
    top_k = 5
    docs = list(get_documents(nr=nr_docs, index_start=0, emb_size=emb_size))
    assert len(docs) == nr_docs

    dump_path = os.path.join(str(tmpdir), 'dump_dir')
    os.environ['STORAGE_WORKSPACE'] = os.path.join(str(tmpdir), 'storage_ws')
    os.environ['QUERY_WORKSPACE'] = os.path.join(str(tmpdir), 'query_ws')
    print(f'STORAGE_WORKSPACE = {os.environ["STORAGE_WORKSPACE"]}')
    print(f'STORAGE_WORKSPACE = {os.environ["QUERY_WORKSPACE"]}')
    os.environ['SHARDS'] = str(shards)
    if shards > 1:
        os.environ['USES_AFTER'] = 'MatchMerger'
    else:
        os.environ['USES_AFTER'] = 'Pass'

    with Flow.load_config(storage_flow_yml) as flow_storage:
        with Flow.load_config(query_flow_yml) as flow_query:
            with TimeContext(f'### indexing {len(docs)} docs'):
                flow_storage.post(on='/index', inputs=docs)

            results = flow_query.post(on='/search', inputs=docs, return_results=True)
            assert len(results[0].docs[0].matches) == 0

            with TimeContext(f'### dumping {len(docs)} docs'):
                flow_storage.post(
                    on='/dump',
                    target_peapod='indexer_storage',
                    parameters={
                        'dump_path': dump_path,
                        'shards': shards,
                        'timeout': -1,
                    },
                )

            dir_size = path_size(dump_path)
            assert dir_size > 0
            print(f'### dump path size: {dir_size} MBs')

            # assert data dumped is correct
            for pea_id in range(shards):
                assert_dump_data(dump_path, docs, shards, pea_id)

            flow_query.rolling_update(pod_name='indexer_query', dump_path=dump_path)

            results = flow_query.post(
                on='/search',
                inputs=docs,
                parameters={'top_k': top_k},
                return_results=True,
            )
            assert len(results[0].docs[0].matches) == top_k
            assert results[0].docs[0].matches[0].scores['similarity'].value == 1.0

    idx = LMDBStorage(
        metas={'workspace': os.environ['STORAGE_WORKSPACE'], 'name': 'lmdb'},
        map_size=1048576000,
        runtime_args={'pea_id': 0},
    )
    assert idx.size == nr_docs
