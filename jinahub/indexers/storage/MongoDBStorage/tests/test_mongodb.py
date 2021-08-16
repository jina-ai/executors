import os
from pathlib import Path

import pytest
import numpy as np
from jina import Document, DocumentArray, Flow, Executor
from jina_commons.indexers.dump import import_vectors, import_metas

from ..mongo_storage import doc_without_embedding

NUM_DOCS = 10
cur_dir = os.path.dirname(os.path.abspath(__file__))
compose_yml = os.path.abspath(os.path.join(cur_dir, 'docker-compose.yml'))


def test_config():
    config_path = Path(__file__).parents[1] / 'config.yml'
    storage = Executor.load_config(str(config_path))
    assert storage._traversal_paths == ['r']


@pytest.mark.parametrize('docker_compose', [compose_yml], indirect=['docker_compose'])
def test_mongo_add(docs_to_index, storage, docker_compose):
    storage.add(docs=docs_to_index, parameters={})
    assert storage.size == NUM_DOCS


@pytest.mark.parametrize('docker_compose', [compose_yml], indirect=['docker_compose'])
def test_mongo_update(docs_to_index, storage, docker_compose):
    storage.add(docs=docs_to_index, parameters={})
    doc_id_to_update = docs_to_index[0].id
    storage.update(
        docs=DocumentArray([Document(id=doc_id_to_update, text='hello test')])
    )
    docs_to_search = DocumentArray([Document(id=doc_id_to_update)])
    storage.search(docs=docs_to_search)
    assert docs_to_search[0].text == 'hello test'


@pytest.mark.parametrize('docker_compose', [compose_yml], indirect=['docker_compose'])
def test_mongo_search(docs_to_index, storage, docker_compose):
    storage.add(docs=docs_to_index, parameters={})
    docs_to_search = DocumentArray([Document(id=docs_to_index[0].id)])
    storage.search(docs=docs_to_search)
    assert docs_to_search[0].text == docs_to_index[0].text


@pytest.mark.parametrize('docker_compose', [compose_yml], indirect=['docker_compose'])
def test_mongo_delete(docs_to_index, storage, docker_compose):
    doc_id_to_delete = docs_to_index[0].id
    storage.delete(docs=DocumentArray([Document(id=doc_id_to_delete)]))
    docs_to_search = DocumentArray([Document(id=doc_id_to_delete)])
    assert len(docs_to_search) == 1
    assert docs_to_search[0].text == ''


@pytest.mark.parametrize('num_shards', [1, 3, 5])
@pytest.mark.parametrize('docker_compose', [compose_yml], indirect=['docker_compose'])
def test_mongo_dump(docs_to_index, storage, tmpdir, num_shards, docker_compose):
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
        f.post(on='/dump', parameters={'dump_path': dump_path, 'shards': num_shards})
    assert os.path.exists(dump_path)
    # recover dumped files and assert identical
    all_ids = []
    all_vecs = []
    all_metas = []
    for pea_id in range(num_shards):
        ids, vecs = import_vectors(dump_path, pea_id=str(pea_id))
        _, metas = import_metas(dump_path, pea_id=str(pea_id))
        all_ids.extend(ids)
        all_vecs.extend(vecs)
        all_metas.extend(metas)
    assert all_ids == docs_to_index.get_attributes('id')
    assert (
        np.asarray(all_vecs) == np.asarray(docs_to_index.get_attributes('embedding'))
    ).all()
    for idx, meta in enumerate(all_metas):
        assert meta == doc_without_embedding(docs_to_index[idx])
    storage.delete(docs_to_index)
