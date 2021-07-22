__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from jina import Flow, Document


def data_generator(num_docs):
    for i in range(num_docs):
        doc = Document(
            text='it is a good day! the dog sits on the floor.')
        yield doc


def test_use_in_flow():
    with Flow.load_config('flow.yml') as flow:
        resp = flow.post(on='/encode', inputs=data_generator(5), return_results=True)
        docs = resp[0].docs
        assert len(docs) == 5
        for doc in docs:
            assert doc.embedding.shape == (768,)
