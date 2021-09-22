from pathlib import Path

import pytest
from doc_cache import DocCache
from jina import Document, DocumentArray, Executor


@pytest.fixture(scope='function')
def cacher(tmp_path) -> DocCache:
    return DocCache(metas={'workspace': str(tmp_path / 'cache')})


def test_config(tmp_path):
    ex = Executor.load_config(
        str(Path(__file__).parents[2] / 'config.yml'),
        override_metas={'workspace': str(tmp_path / 'cache')},
    )
    assert len(ex.fields) == 1
    assert ex.fields[0] == 'content_hash'


def test_cache_crud(cacher):
    docs = DocumentArray(
        [
            Document(id=1, content='content'),
            Document(id=2, content='content'),
            Document(id=3, content='content'),
            Document(id=4, content='content2'),
        ]
    )

    cacher.index_or_remove_from_request(docs)
    # cache all the docs by id and remove the ones that have already been "hit"
    assert cacher.ids_count == 4
    assert cacher.hashes_count == 2

    docs = DocumentArray(
        [
            Document(id=1, content='content3'),
            Document(id=2, content='content4'),
            Document(id=3, content='contentX'),
            Document(id=4, content='contentBLA'),
        ]
    )

    cacher.update(docs)
    assert cacher.ids_count == 4
    assert cacher.hashes_count == 4

    docs = DocumentArray(
        [
            Document(id=1),
            Document(id=2),
            Document(id=3),
            Document(id=4),
            Document(id=4),
            Document(id=5),
            Document(id=6),
            Document(id=7),
        ]
    )

    cacher.delete(docs)
    assert cacher.ids_count == 0
    assert cacher.hashes_count == 0


def test_empty_docs(cacher: DocCache):
    docs = DocumentArray()
    cacher.index_or_remove_from_request(docs=docs)
    assert len(docs) == 0


def test_none_docs(cacher: DocCache):
    try:
        cacher.index_or_remove_from_request(docs=None)
    except Exception:
        pytest.fail('index failed')


def test_docs_no_fields(tmp_path):
    cacher = DocCache(fields=('text',), metas={'workspace': str(tmp_path / 'cache')})

    docs = DocumentArray()
    for i in range(4):
        docs.append(Document(text=f'{i % 2}'))

    # add docs without the `text` field which is required for caching
    import numpy as np

    for _ in range(2):
        docs.append(Document(blob=np.array([1, 2])))

    try:
        cacher.index_or_remove_from_request(docs=docs)
        assert len(docs) == 3
        assert docs[-1].text == ''
        assert docs[-1].blob.shape == (2,)
    except Exception:
        pytest.fail('index failed, docs have no required field')
