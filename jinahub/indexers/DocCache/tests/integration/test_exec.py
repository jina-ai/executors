import os
from itertools import cycle
from pathlib import Path

import pytest
from doc_cache import DocCache
from jina import Document, DocumentArray, Flow

cur_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize('cache_fields', ['[content_hash]', '[id]'])
@pytest.mark.parametrize('value', [['a'], ['a', 'b']])
def test_cache(tmp_path, cache_fields, value):
    os.environ['CACHE_FIELDS'] = cache_fields
    os.environ['CACHE_WORKSPACE'] = str(tmp_path / 'cache')
    docs = DocumentArray()
    for _id, v in enumerate(cycle(value)):
        if _id >= 2:
            break
        if cache_fields == '[content_hash]':
            doc = Document(content=v)
        elif cache_fields == '[id]':
            doc = Document(id=v)
        docs.append(doc)

    with Flow.load_config(os.path.join(cur_dir, 'flow.yml')) as f:
        response = f.post(on='/index', inputs=DocumentArray(docs), return_results=True)
        assert len(response[0].docs) == len(value)
        if cache_fields == '[content_hash]':
            result_set = set([d.content for d in response[0].docs])
        elif cache_fields == '[id]':
            result_set = set([d.id for d in response[0].docs])
        assert result_set == set(value)


@pytest.mark.parametrize('content', [['a'], ['a', 'b']])
def test_cache_two_fields(tmp_path, content):
    os.environ['CACHE_FIELDS'] = '[id, content]'
    os.environ['CACHE_WORKSPACE'] = str(tmp_path / 'cache')
    docs = DocumentArray()
    for _id, c in enumerate(cycle(content)):
        if _id >= 3:
            break
        docs.append(Document(id=c, content='content'))
    with Flow.load_config(os.path.join(cur_dir, 'flow.yml')) as f:
        response = f.post(on='/index', inputs=docs, return_results=True)
        assert len(response[0].docs) == len(set(content))


def test_default_config(tmp_path):
    import shutil

    shutil.rmtree(str(Path(__file__).parents[2] / 'cache'), ignore_errors=True)
    docs = DocumentArray(
        [
            Document(id=1, content='🐯'),
            Document(id=2, content='🐯'),
            Document(id=3, content='🐻'),
        ]
    )

    docs_to_update = DocumentArray([Document(id=2, content='🐼')])

    with Flow().add(
        uses=str(Path(__file__).parents[2] / 'config.yml'),
        uses_metas={'workspace': str(tmp_path / 'cache')},
    ) as f:
        response = f.post(on='/index', inputs=docs, return_results=True)
        # the duplicated Document is removed from the request
        assert len(response[0].data.docs) == 2
        assert set([doc.id for doc in response[0].data.docs]) == set(['1', '3'])

        response = f.post(on='/update', inputs=docs_to_update, return_results=True)
        # the Document with `id=2` is no longer duplicated.
        assert len(response[0].data.docs) == 1

        response = f.post(on='/index', inputs=docs[-1], return_results=True)
        assert len(response[0].data.docs) == 0  # the Document has been cached

        f.post(on='/delete', inputs=docs[-1])
        response = f.post(on='/index', inputs=docs[-1], return_results=True)
        # the Document is cached again after the deletion
        assert len(response[0].data.docs) == 1
