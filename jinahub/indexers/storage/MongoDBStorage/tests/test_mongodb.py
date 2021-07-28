import os
import time

import pytest
import numpy as np
from jina import Document, DocumentArray, Flow
from jina_commons.indexers.dump import import_metas, import_vectors

from .. import MongoDBStorage

cur_dir = os.path.dirname(os.path.abspath(__file__))
compose_yml = os.path.abspath(os.path.join(cur_dir, 'docker-compose.yml'))
num_docs = 10
num_shards = 2


@pytest.fixture
def docker_compose(request):
    os.system(
        f"docker-compose -f {request.param} --project-directory . up  --build -d --remove-orphans"
    )
    time.sleep(5)
    yield
    os.system(
        f"docker-compose -f {request.param} --project-directory . down --remove-orphans"
    )


@pytest.fixture
def docs_to_index():
    docu_array = DocumentArray()
    for idx in range(0, num_docs):
        d = Document(text=f'hello {idx}')
        d.embedding = np.random.random(20)
        docu_array.append(d)
    return docu_array


def _assert_dump_data(dump_path, docs, shards, pea_id):
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


def doc_without_embedding(d: Document):
    new_doc = Document(d, copy=True, hash_content=False)
    new_doc.ClearField('embedding')
    return new_doc.SerializeToString()


@pytest.mark.parametrize('docker_compose', [compose_yml], indirect=['docker_compose'])
def test_mongo_storage(docs_to_index, tmpdir, docker_compose):
    # add
    storage = MongoDBStorage()
    storage.add(docs=docs_to_index, parameters={})
    assert storage.size == num_docs
    # update & search
    doc_id_to_update = docs_to_index[0].id
    storage.update(
        docs=DocumentArray([Document(id=doc_id_to_update, text='hello test')])
    )
    docs_to_search = DocumentArray([Document(id=doc_id_to_update)])
    storage.search(docs=docs_to_search)
    assert docs_to_search[0].text == 'hello test'
    # delete
    doc_id_to_delete = docs_to_index[0].id
    storage.delete(docs=DocumentArray([Document(id=doc_id_to_delete)]))
    docs_to_search = DocumentArray([Document(id=doc_id_to_delete)])
    assert len(docs_to_search) == 1
    assert docs_to_search[0].text == ''  # find no result


@pytest.mark.parametrize('docker_compose', [compose_yml], indirect=['docker_compose'])
def test_dump(docs_to_index, tmpdir, docker_compose):
    metas = {'workspace': str(tmpdir), 'name': 'storage'}
    dump_path = os.path.join(tmpdir, 'dump_dir')

    # indexing
    with Flow().add(
        uses={
            'jtype': 'MongoDBStorage',
            'with': {},
            'metas': metas,
        }
    ) as f:
        f.index(inputs=docs_to_index)
        f.post(on='/dump', parameters={'dump_path': dump_path, 'shards': num_shards})

    for pea_id in range(num_shards):
        _assert_dump_data(dump_path, docs_to_index, num_shards, pea_id)


@pytest.mark.parametrize('docker_compose', [compose_yml], indirect=['docker_compose'])
def test_mwu(docker_compose):
    f = Flow().add(uses=MongoDBStorage)

    with f:
        resp = f.post(
            on='/index', inputs=DocumentArray([Document()]), return_results=True
        )
        print(f'{resp}')
