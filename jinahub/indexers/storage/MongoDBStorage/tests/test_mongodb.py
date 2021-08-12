import os

from jina import Document, DocumentArray, Flow

NUM_DOCS = 10
NUM_SHARDS = 2


def test_mongo_add(docs_to_index, storage):
    storage.add(docs=docs_to_index, parameters={})
    assert storage.size == NUM_DOCS


def test_mongo_update(docs_to_index, storage):
    doc_id_to_update = docs_to_index[0].id
    storage.update(
        docs=DocumentArray([Document(id=doc_id_to_update, text='hello test')])
    )


def test_mongo_search(docs_to_index, storage):
    storage.add(docs=docs_to_index, parameters={})
    docs_to_search = DocumentArray([Document(id=docs_to_index[0].id)])
    storage.search(docs=docs_to_search)
    assert docs_to_search[0].text == docs_to_index[0].text


def test_mongo_delete(docs_to_index, storage):
    doc_id_to_delete = docs_to_index[0].id
    storage.delete(docs=DocumentArray([Document(id=doc_id_to_delete)]))
    docs_to_search = DocumentArray([Document(id=doc_id_to_delete)])
    assert len(docs_to_search) == 1
    assert docs_to_search[0].text == ''


def test_mongo_dump(docs_to_index, storage, tmpdir):
    storage.add(docs=docs_to_index, parameters={})
    metas = {'workspace': str(tmpdir), 'name': 'storage'}
    dump_path = os.path.join(tmpdir, 'dump_dir')
    with Flow().add(
        uses={
            'jtype': 'MongoDBStorage',
            'with': {},
            'metas': metas,
        }
    ) as f:
        f.index(inputs=docs_to_index)
        f.post(on='/dump', parameters={'dump_path': dump_path, 'shards': NUM_SHARDS})
    assert os.path.exists(dump_path)
