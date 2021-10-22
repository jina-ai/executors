import numpy as np
import pytest
from hnswlib_searcher import HnswlibSearcher
from jina import Document, DocumentArray, Flow

_DIM = 10


@pytest.mark.parametrize('uses', ['HnswlibSearcher', 'docker://hnswlibsearcher'])
def test_index_search_flow(uses: str, build_docker_image: str):
    f = Flow().add(uses=uses, uses_with={'metric': 'l2', 'dim': _DIM})
    da = DocumentArray(
        [
            Document(id='a', embedding=np.ones(_DIM) * 1.0),
            Document(id='b', embedding=np.ones(_DIM) * 2.0),
        ]
    )

    with f:
        f.index(da)

        status_ind = f.post('/status', return_results=True)
        status_ind = status_ind[0].data.docs[0].tags

        assert status_ind['count_active'] == 2
        assert status_ind['count_deleted'] == 0
        assert status_ind['count_indexed'] == 2

        result_search = f.search(da, return_results=True)
        result_search = result_search[0].data.docs
        assert len(result_search) == 2

        for ind in range(2):
            assert result_search[ind].matches[0].id == ('a' if ind == 0 else 'b')
            assert result_search[ind].matches[0].scores['l2'].value == 0.0
            assert result_search[ind].matches[1].id == ('b' if ind == 0 else 'a')
            assert result_search[ind].matches[1].scores['l2'].value == 10.0


def test_save_load(tmp_path):
    f = Flow().add(
        name='hnsw', uses=HnswlibSearcher, uses_with={'metric': 'l2', 'dim': _DIM}
    )
    da = DocumentArray(
        [
            Document(id='a', embedding=np.ones(_DIM) * 1.0),
            Document(id='b', embedding=np.ones(_DIM) * 2.0),
        ]
    )

    # Index and save
    with f:
        f.index(da)
        f.post(
            on='/dump',
            target_peapod='hnsw',
            parameters={
                'dump_path': str(tmp_path),
            },
        )

    # Sanity check - without "dump_path" specified, index is empty
    with f:
        status_ind = f.post('/status', return_results=True)
        status_ind = status_ind[0].data.docs[0].tags
        assert status_ind['count_active'] == 0
        assert status_ind['count_indexed'] == 0

    # Load
    f = Flow().add(
        name='hnsw',
        uses=HnswlibSearcher,
        uses_with={'metric': 'l2', 'dim': _DIM, 'dump_path': str(tmp_path)},
    )
    with f:
        status_ind = f.post('/status', return_results=True)
        status_ind = status_ind[0].data.docs[0].tags

        assert status_ind['count_active'] == 2
        assert status_ind['count_deleted'] == 0
        assert status_ind['count_indexed'] == 2

        # Check that we indeed have same items in index
        result_search = f.search(da, return_results=True)
        result_search = result_search[0].data.docs
        assert len(result_search) == 2

        for ind in range(2):
            assert result_search[ind].matches[0].id == ('a' if ind == 0 else 'b')
            assert result_search[ind].matches[0].scores['l2'].value == 0.0
            assert result_search[ind].matches[1].id == ('b' if ind == 0 else 'a')
            assert result_search[ind].matches[1].scores['l2'].value == 10.0


def test_search_limit(tmp_path):
    f = Flow().add(
        name='hnsw', uses=HnswlibSearcher, uses_with={'metric': 'l2', 'dim': _DIM}
    )
    da = DocumentArray(
        [
            Document(id='a', embedding=np.ones(_DIM) * 1.0),
            Document(id='b', embedding=np.ones(_DIM) * 2.0),
        ]
    )

    # Index
    with f:
        f.index(da)

        # Search by specifying limit
        result_search = f.search(da, return_results=True, parameters={'limit': 1})
        for doc in result_search[0].docs:
            assert len(doc.matches) == 1