from jina import DocumentArray, Flow
from jinahub.encoder.clip_text import CLIPTextEncoder

def test_no_documents():
    test_docs = DocumentArray()
    f = Flow().add(uses=CLIPTextEncoder)
    with f:
        f.search(test_docs, {})
    assert len(test_docs) == 0  # SUCCESS
