__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import gzip
import os
from pathlib import Path

import numpy as np
import pytest
from jina import DocumentArray, Document, Executor
from jina.executors.metas import get_default_metas

from jina_commons.indexers.dump import export_dump_streaming
from ..faiss_searcher import FaissSearcher


def _get_docs_from_vecs(queries):
    docs = DocumentArray()
    for q in queries:
        doc = Document(embedding=q)
        docs.append(doc)
    return docs


# fix the seed here
np.random.seed(500)
retr_idx = None
vec_idx = np.random.randint(0, high=512, size=[512]).astype(str)
vec = np.array(np.random.random([512, 10]), dtype=np.float32)

query = np.array(np.random.random([10, 10]), dtype=np.float32)
query_docs = _get_docs_from_vecs(query)

cur_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope='function', autouse=True)
def metas(tmpdir):
    os.environ['TEST_WORKSPACE'] = str(tmpdir)
    metas = get_default_metas()
    metas['workspace'] = os.environ['TEST_WORKSPACE']
    metas['name'] = 'faiss_idx'
    yield metas
    del os.environ['TEST_WORKSPACE']


@pytest.fixture()
def tmpdir_dump(tmpdir):
    from jina_commons.indexers.dump import export_dump_streaming

    export_dump_streaming(
        os.path.join(tmpdir, 'dump'),
        1,
        len(vec_idx),
        zip(vec_idx, vec, [b'' for _ in range(len(vec))]),
    )
    return os.path.join(tmpdir, 'dump')


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[1] / 'config.yml'))
    assert ex.index_key == 'IVF10,PQ4'


def test_faiss_indexer_empty(metas):
    indexer = FaissSearcher(
        index_key='IVF10,PQ2',
        metas=metas,
        runtime_args={'pea_id': 0},
        prefetch_size=256,
    )
    indexer.search(query_docs, parameters={'top_k': 4})
    assert len(query_docs[0].matches) == 0


def test_faiss_indexer(metas, tmpdir_dump):
    import faiss

    trained_index_file = os.path.join(os.environ['TEST_WORKSPACE'], 'faiss.index')
    train_data = np.array(np.random.random([1024, 10]), dtype=np.float32)
    faiss_index = faiss.index_factory(10, 'IVF10,PQ2')
    faiss_index.train(train_data)
    faiss.write_index(faiss_index, trained_index_file)

    indexer = FaissSearcher(
        prefetch_size=256,
        index_key='IVF10,PQ2',
        trained_index_file=trained_index_file,
        dump_path=tmpdir_dump,
        metas=metas,
        runtime_args={'pea_id': 0},
    )
    indexer.search(query_docs, parameters={'top_k': 4})
    assert len(query_docs[0].matches) == 4
    for d in query_docs:
        assert (
            d.matches[0].scores[indexer.metric].value
            >= d.matches[1].scores[indexer.metric].value
        )


@pytest.mark.parametrize('index_key', ['Flat'])
def test_fill_embeddings(index_key, metas, tmpdir_dump):
    indexer = FaissSearcher(
        prefetch_size=256,
        index_key=index_key,
        dump_path=tmpdir_dump,
        metas=metas,
        runtime_args={'pea_id': 0},
    )
    indexer.search(query_docs, parameters={'top_k': 4})
    da = DocumentArray(
        [Document(id=vec_idx[0]), Document(id=vec_idx[1]), Document(id=99999999)]
    )
    indexer.fill_embedding(da)
    assert da[str(vec_idx[0])].embedding is not None
    assert da[str(vec_idx[0])].embedding is not None
    assert da['99999999'].embedding is None


@pytest.mark.parametrize('index_key', ['IVF10,PQ2', 'LSH'])
def test_fill_embeddings_fail(index_key, metas, tmpdir_dump):
    indexer = FaissSearcher(
        prefetch_size=256,
        index_key=index_key,
        dump_path=tmpdir_dump,
        metas=metas,
        runtime_args={'pea_id': 0},
    )
    indexer.search(query_docs, parameters={'top_k': 4})
    da = DocumentArray(
        [Document(id=vec_idx[0]), Document(id=vec_idx[1]), Document(id=99999999)]
    )
    indexer.fill_embedding(da)
    assert da[str(vec_idx[0])].embedding is None
    assert da[str(vec_idx[0])].embedding is None
    assert da['99999999'].embedding is None


@pytest.mark.parametrize(
    ['metric', 'is_distance'],
    [('l2', True), ('inner_product', True), ('l2', False), ('inner_product', False)],
)
def test_faiss_metric(metas, tmpdir_dump, metric, is_distance):
    indexer = FaissSearcher(
        prefetch_size=256,
        index_key='IVF10,PQ2',
        metric=metric,
        is_distance=is_distance,
        dump_path=tmpdir_dump,
        metas=metas,
        runtime_args={'pea_id': 0},
    )
    query = np.array(np.random.random([10, 10]), dtype=np.float32)
    docs = _get_docs_from_vecs(query)
    indexer.search(docs, parameters={'top_k': 4})
    assert len(docs[0].matches) == 4

    for i in range(len(docs[0].matches) - 1):
        if not is_distance:
            assert (
                docs[0].matches[i].scores[metric].value
                >= docs[0].matches[i + 1].scores[metric].value
            )
        else:
            assert (
                docs[0].matches[i].scores[metric].value
                <= docs[0].matches[i + 1].scores[metric].value
            )


def test_faiss_indexer_known(metas, tmpdir):
    vectors = np.array(
        [[1, 1, 1], [10, 10, 10], [100, 100, 100], [1000, 1000, 1000]], dtype=np.float32
    )
    keys = np.array([4, 5, 6, 7]).astype(str)
    export_dump_streaming(
        os.path.join(tmpdir, 'dump'),
        1,
        len(keys),
        zip(keys, vectors, [b'' for _ in range(len(vectors))]),
    )

    indexer = FaissSearcher(
        prefetch_size=256,
        index_key='Flat',
        metas=metas,
        dump_path=os.path.join(tmpdir, 'dump'),
        runtime_args={'pea_id': 0},
    )
    assert indexer.size == len(keys)

    queries = np.array(
        [[1, 1, 1], [10, 10, 10], [100, 100, 100], [1000, 1000, 1000]], dtype=np.float32
    )
    TOP_K = 2
    docs = _get_docs_from_vecs(queries)
    indexer.search(docs, parameters={'top_k': TOP_K})
    idx = docs.traverse_flat(['m']).get_attributes('id')
    dist = docs.traverse_flat(['m']).get_attributes('scores')
    np.testing.assert_equal(
        idx, np.concatenate(np.array([[4, 5], [5, 4], [6, 5], [7, 6]])).astype(str)
    )
    assert len(idx) == len(dist)
    assert len(idx) == len(docs) * TOP_K


def test_faiss_indexer_known_big(metas, tmpdir):
    """Let's try to have some real test. We will have an index with 10k vectors of random values between 5 and 10.
    We will change tweak some specific vectors that we expect to be retrieved at query time. We will tweak vector
    at index [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000], this will also be the query vectors.
    Then the keys will be assigned shifted to test the proper usage of `int2ext_id` and `ext2int_id`
    """
    vectors = np.random.uniform(low=5.0, high=10.0, size=(10000, 1024)).astype(
        'float32'
    )

    queries = np.empty((10, 1024), dtype=np.float32)
    for idx in range(0, 10000, 1000):
        array = idx * np.ones((1, 1024), dtype=np.float32)
        queries[int(idx / 1000)] = array
        vectors[idx] = array

    keys = np.arange(10000, 20000).astype(str)

    dump_path = os.path.join(tmpdir, 'dump')
    export_dump_streaming(
        dump_path,
        1,
        len(keys),
        zip(keys, vectors, [b'' for _ in range(len(vectors))]),
    )
    indexer = FaissSearcher(
        prefetch_size=256,
        index_key='Flat',
        requires_training=True,
        metas=metas,
        dump_path=dump_path,
        runtime_args={'pea_id': 0},
    )
    assert isinstance(indexer, FaissSearcher)
    docs = _get_docs_from_vecs(queries)
    top_k = 1
    indexer.search(docs, parameters={'top_k': top_k})
    idx = docs.traverse_flat(['m']).get_attributes('id')
    np.testing.assert_equal(
        idx,
        np.concatenate(
            np.array(
                [
                    [10000],
                    [11000],
                    [12000],
                    [13000],
                    [14000],
                    [15000],
                    [16000],
                    [17000],
                    [18000],
                    [19000],
                ]
            )
        ).astype(str),
    )
    dist = docs.traverse_flat(['m']).get_attributes('scores')
    assert len(idx) == len(dist)
    assert len(idx) == (10 * top_k)


@pytest.mark.parametrize('train_data', ['new', 'none'])
@pytest.mark.parametrize('max_num_points', [None, 257, 500, 10000])
def test_indexer_train(metas, train_data, max_num_points, tmpdir):
    np.random.seed(500)
    num_data = 500
    num_dim = 64
    num_query = 10
    query = np.array(np.random.random([num_query, num_dim]), dtype=np.float32)
    vec_idx = np.random.randint(0, high=num_data, size=[num_data]).astype(str)
    vec = np.random.random([num_data, num_dim])

    dump_path = os.path.join(tmpdir, 'dump')
    export_dump_streaming(
        dump_path,
        1,
        len(vec_idx),
        zip(vec_idx, vec, [b'' for _ in range(len(vec))]),
    )
    indexer = FaissSearcher(
        prefetch_size=256,
        index_key='IVF10,PQ4',
        max_num_training_points=max_num_points,
        requires_training=True,
        metas=metas,
        dump_path=dump_path,
        runtime_args={'pea_id': 0},
    )

    query_docs = _get_docs_from_vecs(query)
    top_k = 4
    indexer.search(query_docs, parameters={'top_k': top_k})
    # idx, dist =
    idx = query_docs.traverse_flat(['m']).get_attributes('id')
    dist = query_docs.traverse_flat(['m']).get_attributes('scores')

    assert len(idx) == len(dist)
    assert len(idx) == num_query * top_k


@pytest.mark.parametrize('metric', ['l2', 'inner_product'])
def test_faiss_normalization(metas, metric, tmpdir):
    num_data = 2
    num_dims = 64

    vecs = np.zeros((num_data, num_dims))
    vecs[:, 0] = 2
    vecs[0, 1] = 3
    keys = np.arange(0, num_data).astype(str)

    dump_path = os.path.join(tmpdir, 'dump')
    export_dump_streaming(
        dump_path,
        1,
        len(keys),
        zip(keys, vecs, [b'' for _ in range(len(vecs))]),
    )

    indexer = FaissSearcher(
        prefetch_size=256,
        index_key='Flat',
        metric=metric,
        normalize=True,
        requires_training=True,
        metas=metas,
        dump_path=dump_path,
        runtime_args={'pea_id': 0},
    )
    query = np.zeros((1, num_dims))
    query[0, 0] = 5
    docs = _get_docs_from_vecs(query.astype('float32'))
    indexer.search(docs, parameters={'top_k': 2})
    dist = docs.traverse_flat(['m']).get_attributes('scores')
    assert dist[0][metric].value == 1.0


@pytest.mark.parametrize('max_num_points', [257, 500, None])
def test_faiss_indexer_train(metas, tmpdir, max_num_points):
    train_filepath = os.path.join(os.environ['TEST_WORKSPACE'], 'train.npy')
    train_data = np.array(np.random.random([1024, 10]), dtype=np.float32)
    np.save(train_filepath, train_data)

    trained_index_file = os.path.join(tmpdir, 'faiss.index')
    indexer = FaissSearcher(
        index_key='IVF10,PQ2',
        trained_index_file=trained_index_file,
        metas=metas,
        runtime_args={'pea_id': 0},
        prefetch_size=256,
    )
    indexer.train(
        parameters={
            'train_filepath': train_filepath,
            'max_num_training_points': max_num_points,
        }
    )
    assert indexer._faiss_index.is_trained


def test_faiss_train_and_index(metas, tmpdir, tmpdir_dump):
    train_filepath = os.path.join(os.environ['TEST_WORKSPACE'], 'train.npy')
    train_data = np.array(np.random.random([1024, 10]), dtype=np.float32)
    np.save(train_filepath, train_data)

    trained_index_file = os.path.join(tmpdir, 'faiss.index')
    indexer = FaissSearcher(
        index_key='IVF10,PQ2',
        trained_index_file=trained_index_file,
        metas=metas,
        runtime_args={'pea_id': 0},
        prefetch_size=256,
    )
    indexer.train(
        parameters={
            'train_filepath': train_filepath,
        }
    )

    trained_indexer = FaissSearcher(
        prefetch_size=256,
        index_key='IVF10,PQ2',
        trained_index_file=trained_index_file,
        dump_path=tmpdir_dump,
        metas=metas,
        runtime_args={'pea_id': 0},
    )
    trained_indexer.search(query_docs, parameters={'top_k': 4})
    assert len(query_docs[0].matches) == 4
    for d in query_docs:
        assert (
            d.matches[0].scores[indexer.metric].value
            >= d.matches[1].scores[indexer.metric].value
        )


def test_faiss_train_before_index(metas, tmpdir, tmpdir_dump):
    indexer = FaissSearcher(
        prefetch_size=256,
        index_key='IVF10,PQ2',
        dump_path=tmpdir_dump,
        metas=metas,
        runtime_args={'pea_id': 0},
    )
    indexer.search(query_docs, parameters={'top_k': 4})
    assert len(query_docs[0].matches) == 4
    for d in query_docs:
        assert (
            d.matches[0].scores[indexer.metric].value
            >= d.matches[1].scores[indexer.metric].value
        )
