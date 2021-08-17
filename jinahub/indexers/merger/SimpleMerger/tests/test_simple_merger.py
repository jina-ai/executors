__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import pytest
from jina import Flow, Document, requests, DocumentArray
from jina.executors import BaseExecutor

from jinahub.indexers.merger.SimpleMerger.simple_merger import SimpleMerger


class MockShard(BaseExecutor):
    @requests
    def search(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            doc.matches.append(Document(tags={'shard_id': self.runtime_args.pea_id}))

@pytest.fixture
def docs():
    return [Document(text=f'sample text {i}') for i in range(2)]

@pytest.mark.parametrize('shards', (1, 3, 5))
def test_simple_merger(docs, shards):
    def callback(resp):
        assert len(resp.docs) == 2
        for doc in resp.docs:
            assert {d.tags['shard_id'] for d in doc.matches} == {float(i) for i in range(shards)}

    with Flow().add(
            uses=MockShard,
            uses_after=SimpleMerger,
            shards=shards,
            polling='all'
    ) as f:
        f.search(docs, on_done=callback)
