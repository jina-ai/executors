from jina import Document, DocumentArray, Flow
from spell_checker import SpellChecker


def test_spell_check_integration(incorrect_text, correct_text, input_training_data):

    train_docs = DocumentArray([Document(content=t) for t in input_training_data])
    with Flow().add(uses=SpellChecker) as f:
        f.post(on='/train', inputs=train_docs)
        input_docs = DocumentArray([Document(content=t) for t in incorrect_text])
        results = f.post(on='/index', inputs=input_docs, return_results=True)
        result_docs = results[0].docs
        assert len(input_docs) == len(incorrect_text)
        for crafted_doc, expected in zip(result_docs, correct_text):
            assert crafted_doc.content == expected
