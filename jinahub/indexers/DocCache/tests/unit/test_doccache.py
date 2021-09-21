from pathlib import Path

from doc_cache import DocCache
from jina import Document, DocumentArray, Executor


def test_config(tmp_path):
    Executor.load_config(
        str(Path(__file__).parents[2] / 'config.yml'),
        override_metas={'workspace': str(tmp_path / 'cache')},
    )


def test_cache_crud(tmp_path):
    docs = DocumentArray(
        [
            Document(id=1, content='content'),
            Document(id=2, content='content'),
            Document(id=3, content='content'),
            Document(id=4, content='content2'),
        ]
    )

    cache = DocCache(
        fields=('content_hash',), metas={'workspace': str(tmp_path / 'cache')}
    )
    cache.index_or_remove_from_request(docs)
    # cache all the docs by id and remove the ones that have already been "hit"
    assert cache.ids_count == 4
    assert cache.hashes_count == 2

    docs = DocumentArray(
        [
            Document(id=1, content='content3'),
            Document(id=2, content='content4'),
            Document(id=3, content='contentX'),
            Document(id=4, content='contentBLA'),
        ]
    )

    cache.update(docs)
    assert cache.ids_count == 4
    assert cache.hashes_count == 4

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

    cache.delete(docs)
    assert cache.ids_count == 0
    assert cache.hashes_count == 0
