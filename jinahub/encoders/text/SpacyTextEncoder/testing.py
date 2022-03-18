from spacy_text_encoder import SpacyTextEncoder
from jina import Flow
from jina import Document, DocumentArray

flow = (
    Flow()
    .add(uses=SpacyTextEncoder, uses_with={"model_name": "en_core_web_md"})
    .add(
        uses="jinahub://SimpleIndexer",
        uses_with={"table_name": "workspace"},
        install_requirements=True,
    )
)

docs = DocumentArray(
    [
        Document(text="a place to drink alcohol"),
        Document(text="willy wonka runs a chocolate factory"),
        Document(text="detective pikachu has a bad day"),
    ]
)

with flow:
    response = flow.index(docs, return_results=True)

print(response)

for doc in response:
    print(doc.embedding)

query_doc = Document(text="I really want a cold beer")

with flow:
    response = flow.search(query_doc, return_results=True)

for doc in response[0].matches:
    print(doc.text)
