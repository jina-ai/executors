from pathlib import Path

import numpy as np
import pytest
from hnswlib_searcher import HnswlibSearcher
from jina import Document, DocumentArray, Executor

_DIM = 10


@pytest.fixture
def two_elem_index():
    index = HnswlibSearcher(dim=_DIM, metric='l2')
    da = DocumentArray(
        [
            Document(id='a', embedding=np.ones(_DIM) * 1.0),
            Document(id='b', embedding=np.ones(_DIM) * 2.0),
        ]
    )
    index.index(da, {})

    return index, da


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[2] / 'config.yml'))
    assert ex.metric == 'cosine'


def test_empty_search():
    index = HnswlibSearcher(dim=_DIM)
    da = DocumentArray([Document(embedding=np.random.normal(size=(_DIM,)))])

    index.search(da, {})
    assert len(da[0].matches) == 0


def test_index_no_docs():
    index = HnswlibSearcher(dim=_DIM)
    index.index(None, {})


def test_index_empty_docs():
    index = HnswlibSearcher(dim=_DIM)
    da = DocumentArray()
    index.index(da, {})


def test_update_no_docs():
    index = HnswlibSearcher(dim=_DIM)
    index.update(None, {})


def test_update_empty_docs():
    index = HnswlibSearcher(dim=_DIM)
    da = DocumentArray()
    index.update(da, {})


def test_search_no_docs():
    index = HnswlibSearcher(dim=_DIM)
    index.search(None, {})


def test_searh_empty_docs():
    index = HnswlibSearcher(dim=_DIM)
    da = DocumentArray()
    index.search(da, {})


def test_index():
    NUM_DOCS = 1000
    index = HnswlibSearcher(dim=_DIM)
    embeddings = np.random.normal(size=(NUM_DOCS, _DIM))
    da1 = DocumentArray([Document(embedding=emb) for emb in embeddings])
    da2 = DocumentArray([Document(embedding=emb) for emb in embeddings])

    index.index(da1, {})
    assert len(index._ids_to_inds) == NUM_DOCS
    assert index._index.element_count == NUM_DOCS
    assert set(index._ids_to_inds.keys()) == set(da1.get_attributes('id'))

    index.index(da2, {})
    assert len(index._ids_to_inds) == 2 * NUM_DOCS
    assert index._index.element_count == 2 * NUM_DOCS
    assert set(index._ids_to_inds.keys()) == set(da1.get_attributes('id')).union(
        da2.get_attributes('id')
    )


def test_index_with_update(two_elem_index):
    index, da = two_elem_index
    da_search = DocumentArray(
        [
            Document(embedding=np.ones(_DIM) * 1.1),
            Document(embedding=np.ones(_DIM) * 2.1),
        ]
    )
    # switch embeddings of a and b
    da[0].embedding = np.ones(_DIM) * 2.0
    da[1].embedding = np.ones(_DIM) * 1.0

    index.index(da, {})
    assert index._ids_to_inds == {'a': 0, 'b': 1}
    assert index._index.element_count == 2

    index.search(da_search, {})
    assert [m.id for m in da_search[0].matches] == ['b', 'a']
    assert [m.id for m in da_search[1].matches] == ['a', 'b']


def test_index_wrong_dim():
    index = HnswlibSearcher(dim=10)
    embeddings = np.random.normal(size=(2, 11))
    da1 = DocumentArray([Document(embedding=emb) for emb in embeddings])

    with pytest.raises(ValueError, match='Attempted to index'):
        index.index(da1, {})


@pytest.mark.parametrize('limit', [5, 10])
@pytest.mark.parametrize(
    ['metric', 'is_distance'],
    [
        ('cosine', True),
        ('euclidean', True),
        ('inner_product', True),
        ('cosine', False),
        ('euclidean', False),
        ('inner_product', False),
    ],
)
def test_search_basic(limit: int, metric: str, is_distance: bool):
    index = HnswlibSearcher(
        dim=_DIM, metric=metric, limit=limit, is_distance=is_distance
    )
    embeddings_ind = np.random.normal(size=(1000, _DIM))
    embeddings_search = np.random.normal(size=(10, _DIM))
    da_index = DocumentArray([Document(embedding=emb) for emb in embeddings_ind])
    da_search = DocumentArray([Document(embedding=emb) for emb in embeddings_search])

    index.index(da_index, {})
    index.search(da_search, {})

    indexed_ids = da_index.get_attributes('id')

    for d in da_search:
        ms = d.matches
        scores = [m.scores[metric].value for m in ms]
        assert len(ms) == limit
        for m in ms:
            assert m.id in indexed_ids

        for i in range(len(ms) - 1):
            if not is_distance:
                assert ms[i].scores[metric].value >= ms[i + 1].scores[metric].value
            else:
                assert ms[i].scores[metric].value <= ms[i + 1].scores[metric].value


def test_topk_max():
    """Test that even with limit set to more than size of index, at most size of
    index elements are returned"""
    index = HnswlibSearcher(dim=_DIM, limit=1000)
    embeddings = np.random.normal(size=(10, _DIM))
    da = DocumentArray([Document(embedding=emb) for emb in embeddings])

    index.index(da, {})
    index.search(da, {})

    for d in da:
        assert len(d.matches) == 10


def test_search_quality():
    """Test that we get everything correct for a small index"""
    index = HnswlibSearcher(dim=_DIM, metric='euclidean')
    da = DocumentArray(
        [
            Document(id='a', embedding=np.ones(_DIM) * 1.1),
            Document(id='b', embedding=np.ones(_DIM) * 2.0),
            Document(id='c', embedding=np.ones(_DIM) * 4.0),
            Document(id='d', embedding=np.ones(_DIM) * 7.0),
            Document(id='e', embedding=np.ones(_DIM) * 11.0),
        ]
    )
    index.index(da, {})
    index.search(da)

    matches_a = [m.id for m in da[0].matches]
    matches_b = [m.id for m in da[1].matches]
    matches_c = [m.id for m in da[2].matches]
    matches_d = [m.id for m in da[3].matches]
    matches_e = [m.id for m in da[4].matches]

    assert matches_a == ['a', 'b', 'c', 'd', 'e']
    assert matches_b == ['b', 'a', 'c', 'd', 'e']
    assert matches_c == ['c', 'b', 'a', 'd', 'e']
    assert matches_d == ['d', 'c', 'e', 'b', 'a']
    assert matches_e == ['e', 'd', 'c', 'b', 'a']

    for doc in da:
        assert doc.matches[0].scores['euclidean'].value == 0


def test_search_wrong_dim():
    index = HnswlibSearcher(dim=_DIM)
    embeddings_ind = np.random.normal(size=(1000, _DIM))
    embeddings_search = np.random.normal(size=(10, 17))
    da_index = DocumentArray([Document(embedding=emb) for emb in embeddings_ind])
    da_search = DocumentArray([Document(embedding=emb) for emb in embeddings_search])

    index.index(da_index, {})

    with pytest.raises(ValueError, match='Query documents have embeddings'):
        index.search(da_search, {})


def test_update(two_elem_index):
    index, da = two_elem_index
    da_search = DocumentArray(
        [
            Document(embedding=np.ones(_DIM) * 1.1),
            Document(embedding=np.ones(_DIM) * 2.1),
        ]
    )
    assert index._ids_to_inds == {'a': 0, 'b': 1}

    index.search(da_search, {})
    assert [m.id for m in da_search[0].matches] == ['a', 'b']
    assert [m.id for m in da_search[1].matches] == ['b', 'a']
    for d in da_search:
        d.pop('matches')

    # switch embeddings of a and b
    da[0].embedding = np.ones(_DIM) * 2.0
    da[1].embedding = np.ones(_DIM) * 1.0

    index.update(da, {})
    assert index._ids_to_inds == {'a': 0, 'b': 1}
    assert index._index.element_count == 2

    index.search(da_search, {})
    assert [m.id for m in da_search[0].matches] == ['b', 'a']
    assert [m.id for m in da_search[1].matches] == ['a', 'b']


def test_update_ignore_non_existing(two_elem_index):
    index, da = two_elem_index
    da_search = DocumentArray(
        [
            Document(embedding=np.ones(_DIM) * 1.1),
            Document(embedding=np.ones(_DIM) * 2.1),
        ]
    )

    # switch embeddings of a and b, and add a new element - it should not get indexed
    da[0].embedding = np.ones(_DIM) * 2.0
    da[1].embedding = np.ones(_DIM) * 1.0
    da.append(Document(id='c', embedding=np.ones(_DIM) * 3.0))

    index.update(da, {})
    assert index._ids_to_inds == {'a': 0, 'b': 1}
    assert index._index.element_count == 2

    index.search(da_search, {})
    assert [m.id for m in da_search[0].matches] == ['b', 'a']
    assert [m.id for m in da_search[1].matches] == ['a', 'b']


def test_update_wrong_dim():
    index = HnswlibSearcher(dim=_DIM)
    embeddings_ind = np.random.normal(size=(10, _DIM))
    embeddings_update = np.random.normal(size=(10, 17))
    da_index = DocumentArray([Document(embedding=emb) for emb in embeddings_ind])

    index.index(da_index, {})

    for doc, emb in zip(da_index, embeddings_update):
        doc.embedding = emb
    with pytest.raises(ValueError, match='Attempted to update vectors with dimension'):
        index.update(da_index, {})


def test_delete(two_elem_index):
    index, da = two_elem_index

    index.delete({'ids': ['a', 'c']})
    assert index._ids_to_inds == {'b': 1}

    index.search(da, {'limit': 10})
    assert len(da[0].matches) == 1


def test_delete_soft(two_elem_index):
    """Test that we do not overwrite deleted indices"""
    index, da = two_elem_index
    assert index._ids_to_inds == {'a': 0, 'b': 1}

    index.delete({'ids': ['b']})
    assert index._ids_to_inds == {'a': 0}
    assert index._index.element_count == 2

    index.index(da[1:2], {})
    assert index._ids_to_inds == {'a': 0, 'b': 2}
    assert index._index.element_count == 3


def test_clear(two_elem_index):
    index, _ = two_elem_index

    index.clear()
    assert len(index._ids_to_inds) == 0
    assert index._index.element_count == 0


def test_dump(two_elem_index, tmp_path):
    index, da = two_elem_index
    index.dump({'dump_path': str(tmp_path)})

    assert (tmp_path / 'index.bin').is_file()
    assert (tmp_path / 'ids.json').is_file()


def test_dump_no_path(two_elem_index):
    index, _ = two_elem_index

    with pytest.raises(ValueError, match='The `dump_path` must be provided'):
        index.dump()


def test_dump_load(tmp_path, two_elem_index):
    index, da = two_elem_index
    index.dump({'dump_path': str(tmp_path)})

    index = HnswlibSearcher(dim=_DIM, metric='l2', dump_path=tmp_path)

    assert index._ids_to_inds == {'a': 0, 'b': 1}
    assert index._index.element_count == 2

    index.search(da, {})
    assert da[0].matches.get_attributes('id') == ['a', 'b']
    assert da[1].matches.get_attributes('id') == ['b', 'a']


def test_status(two_elem_index):
    index, _ = two_elem_index
    status = index.status()[0]

    assert status.tags['count_active'] == 2
    assert status.tags['count_indexed'] == 2
    assert status.tags['count_deleted'] == 0

    index.delete({'ids': ['a']})
    status = index.status()[0]

    assert status.tags['count_active'] == 1
    assert status.tags['count_indexed'] == 2
    assert status.tags['count_deleted'] == 1
