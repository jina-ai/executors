__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from jina import Flow

from ...minranker import MinRanker


def test_integration(documents_chunk):
    with Flow().add(uses=MinRanker, uses_with={'metric': 'cosine'}) as flow:
        resp = flow.post(on='/search', inputs=documents_chunk, return_results=True)

    for r in resp:
        for doc in r.docs:
            assert doc.matches
            for i in range(len(doc.matches) - 1):
                match = doc.matches[i]
                assert match.tags
                assert (
                    match.scores['cosine'].value
                    >= doc.matches[i + 1].scores['cosine'].value
                )
