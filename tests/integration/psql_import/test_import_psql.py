import collections
import datetime
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
from jina import Document, DocumentArray, Executor, Flow, requests
from jina.logging.profile import TimeContext
from jina_commons.indexers.dump import import_metas, import_vectors

from jinahub.indexers.searcher.compound.FaissPostgresIndexer import FaissPostgresIndexer
from jinahub.indexers.storage.PostgreSQLStorage.postgres_indexer import (
    PostgreSQLStorage,
)
from jinahub.indexers.storage.PostgreSQLStorage.postgreshandler import (
    doc_without_embedding,
)

METRIC = 'l2'


def _flow(uses_after, total_shards, startup_args, polling, replicas=1, name='indexer'):
    return Flow().add(
        name=name,
        uses=FaissPostgresIndexer,
        uses_with={
            'startup_sync_args': startup_args,
        },
        uses_metas={'name': name},
        parallel=total_shards,
        replicas=replicas,
        polling=polling,
        uses_after=uses_after,
    )


@pytest.fixture()
def docker_compose(request):
    os.system(
        f'docker-compose -f {request.param} --project-directory . up  --build -d '
        f'--remove-orphans'
    )
    time.sleep(5)
    yield
    os.system(
        f'docker-compose -f {request.param} --project-directory . down '
        f'--remove-orphans'
    )


cur_dir = os.path.dirname(os.path.abspath(__file__))
compose_yml = os.path.join(cur_dir, 'docker-compose.yml')


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
                try:
                    doc.matches = sorted(
                        doc.matches,
                        key=lambda m: m.scores['cosine'].value,
                        reverse=True,
                    )[:top_k]
                except TypeError as e:
                    print(f'##### {e}')

            docs = DocumentArray(list(results.values()))
            return docs


def get_documents(nr=10, index_start=0, emb_size=7):
    random_batch = np.random.random([nr, emb_size]).astype(np.float32)
    for i in range(index_start, nr + index_start):
        d = Document()
        d.id = f'aa{i}'  # to test it supports non-int ids
        d.embedding = random_batch[i - index_start]
        yield d


def get_batch_iterator(batches, batch_size, emb_size=7):
    index_start = 0
    for batch in range(batches):
        yield from get_documents(batch_size, index_start, emb_size)
        index_start += batch_size


def assert_dump_data(dump_path, docs, shards, pea_id):
    docs = sorted(
        docs, key=lambda doc: doc.id
    )  # necessary since the ordering is done as str in PSQL
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

    # TODO these might fail if we implement any ordering of elements on dumping /
    #  reloading
    ids_dump = list(ids_dump)
    vectors_dump = list(vectors_dump)
    np.testing.assert_equal(set(ids_dump), set([d.id for d in docs_expected]))
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


def flatten(it):
    for x in it:
        if isinstance(x, collections.Iterable) and not isinstance(x, str):
            yield from flatten(x)
        else:
            yield x


# replicas w 1 shard doesn't work
# TODO /sync doesn't work with replicas
@pytest.mark.parametrize('shards', [1, 3, 7])
@pytest.mark.parametrize('nr_docs', [100])
@pytest.mark.parametrize('emb_size', [10])
@pytest.mark.parametrize('replicas', [1])
@pytest.mark.parametrize('snapshot', [True, False])
@pytest.mark.parametrize('docker_compose', [compose_yml], indirect=['docker_compose'])
def test_psql_import(
    tmpdir,
    nr_docs,
    emb_size,
    shards,
    replicas,
    docker_compose,
    snapshot: bool,
    benchmark=False,
):
    top_k = 50
    batch_size = min(1000, nr_docs)
    docs = get_batch_iterator(
        batches=nr_docs // batch_size, batch_size=batch_size, emb_size=emb_size
    )
    if not benchmark:
        docs = DocumentArray(flatten(docs))

    os.environ['STORAGE_WORKSPACE'] = os.path.join(str(tmpdir), 'index_ws')
    os.environ['SHARDS'] = str(shards)

    # we only need one Flow
    # but we make two because
    # of polling
    storage_shards = shards
    if benchmark:
        storage_shards //= 2

    with _flow(
        uses_after='Pass',
        total_shards=storage_shards,
        startup_args={},
        polling='any',
        name='indexer_storage',
    ) as flow_storage:
        with _flow(
            uses_after='MatchMerger',
            total_shards=shards,
            startup_args={},
            polling='all',
            replicas=replicas,
            name='indexer_query',
        ) as flow_query:
            # necessary since PSQL instance might not have shutdown properly
            # between tests
            if not benchmark:
                flow_storage.post(on='/delete', inputs=docs)

            with TimeContext(f'### indexing {nr_docs} docs'):
                flow_storage.post(on='/index', inputs=docs)

            results = flow_query.post(
                on='/search',
                inputs=get_documents(nr=1, index_start=0, emb_size=emb_size),
                return_results=True,
            )
            assert len(results[0].docs[0].matches) == 0

            if snapshot:
                with TimeContext(f'### snapshotting {nr_docs} docs'):
                    flow_storage.post(
                        on='/snapshot',
                    )

            with TimeContext(f'### importing {nr_docs} docs'):
                flow_query.post(on='/sync')

            params = {'top_k': nr_docs}
            if benchmark:
                params = {'top_k': top_k}
            results = flow_query.post(
                on='/search',
                inputs=get_documents(nr=3, index_start=0, emb_size=emb_size),
                parameters=params,
                return_results=True,
            )
            if benchmark:
                assert len(results[0].docs[0].matches) == top_k
            else:
                assert len(results[0].docs[0].matches) == nr_docs
            # TODO score is not deterministic
            assert results[0].docs[0].matches[0].scores[METRIC].value > 0.0

    idx = PostgreSQLStorage()
    assert idx.size == nr_docs


def _in_docker():
    """Returns: True if running in a Docker container, else False"""
    with open('/proc/1/cgroup', 'rt') as ifh:
        if 'docker' in ifh.read():
            print('in docker, skipping benchmark')
            return True
        return False


@pytest.mark.parametrize('snapshot', [True, False])
@pytest.mark.parametrize('docker_compose', [compose_yml], indirect=['docker_compose'])
def test_benchmark(tmpdir, snapshot, docker_compose):
    # benchmark only
    nr_docs = 1000000
    if _in_docker() or ('GITHUB_WORKFLOW' in os.environ):
        nr_docs = 1000
    return test_psql_import(
        tmpdir,
        snapshot=snapshot,
        nr_docs=nr_docs,
        emb_size=256,
        shards=2 * 2,  # to make up for replicas, in comparison
        # TODO make sync work with replicas.
        #  Replicas have polling `any`
        #  and sync needs all
        replicas=1,
        docker_compose=compose_yml,
        benchmark=True,
    )


@pytest.mark.parametrize('docker_compose', [compose_yml], indirect=['docker_compose'])
def test_start_up(docker_compose):
    docs = list(get_documents(nr=100, index_start=0, emb_size=10))
    shards = 1

    with _flow(
        uses_after='MatchMerger', total_shards=shards, startup_args={}, polling='any'
    ) as flow:
        flow.post(on='/index', inputs=docs)
        results = flow.post(
            on='/search',
            inputs=docs,
            parameters={'top_k': len(docs)},
            return_results=True,
        )
        assert len(results[0].docs[0].matches) == 0

        flow.post(
            on='/sync',
        )

        results = flow.post(
            on='/search',
            inputs=docs,
            parameters={'top_k': len(docs)},
            return_results=True,
        )
        assert len(results[0].docs[0].matches) == len(docs)

        flow.delete(inputs=docs)
        # not synced, data is still there in Faiss
        assert len(results[0].docs[0].matches) == len(docs)

        flow.post(
            on='/sync',
        )
        results = flow.post(
            on='/search',
            inputs=docs,
            parameters={'top_k': len(docs)},
            return_results=True,
        )
        assert len(results[0].docs[0].matches) == 0


@pytest.mark.parametrize('shards', [1])
@pytest.mark.parametrize('nr_docs', [100])
@pytest.mark.parametrize('emb_size', [10])
@pytest.mark.parametrize('docker_compose', [compose_yml], indirect=['docker_compose'])
def test_psql_sync_delta(
    tmpdir,
    nr_docs,
    emb_size,
    shards,
    docker_compose,
):
    batch_size = min(1000, nr_docs)
    docs_original = get_batch_iterator(
        batches=nr_docs // batch_size, batch_size=batch_size, emb_size=emb_size
    )
    docs_original = DocumentArray(flatten(docs_original))
    assert len(docs_original) == nr_docs

    os.environ['STORAGE_WORKSPACE'] = os.path.join(str(tmpdir), 'index_ws')
    os.environ['SHARDS'] = str(shards)
    uses_after = 'Pass'

    with _flow(
        uses_after=uses_after, total_shards=shards, startup_args={}, polling='all'
    ) as flow:
        flow.post(on='/index', inputs=docs_original)

        # we first sync by snapshot and then by delta
        flow.post(
            on='/snapshot',
        )

        flow.post(
            on='/sync',
        )

        # delta syncing
        # we delete some old data
        nr_docs_to_delete = 90
        flow.post(on='/delete', inputs=docs_original[:nr_docs_to_delete])
        # update rest of the data with perfect matches of
        docs_search = DocumentArray(
            list(
                get_documents(
                    nr_docs - nr_docs_to_delete,
                    index_start=nr_docs_to_delete,
                    emb_size=emb_size,
                )
            )
        )

        flow.post(on='/update', inputs=docs_search)
        # call sync with delta
        flow.post(on='/sync', parameters={'use_delta': True})
        results = flow.post(
            on='/search',
            inputs=docs_search,
            parameters={'top_k': len(docs_search)},
            return_results=True,
        )
        # then we assert the contents include the latest
        # perfect matches
        assert len(results[0].docs) > 0
        for d in results[0].docs:
            np.testing.assert_almost_equal(d.matches[0].embedding, d.embedding)

    idx = PostgreSQLStorage()
    # size stays the same because it was only soft delete
    assert idx.size == nr_docs
