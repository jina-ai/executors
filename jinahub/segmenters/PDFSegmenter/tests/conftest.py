import os

import pytest
from jina import Document, DocumentArray


@pytest.fixture()
def test_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def expected_text():
    expected_text = (
        "A cat poem\nI love cats, I love every kind of cat,\nI just wanna hug all of them, but I can't,"
        "\nI'm thinking about cats again\nI think about how cute they are\nAnd their whiskers and their "
        "nose"
    )
    return expected_text


@pytest.fixture
def input_pdf(test_dir: str):
    path_img_text = os.path.join(test_dir, 'data/cats_are_awesome.pdf')
    path_text = os.path.join(test_dir, 'data/cats_are_awesome_text.pdf')
    path_img = os.path.join(test_dir, 'data/cats_are_awesome_img.pdf')

    with open(path_text, 'rb') as pdf:
        input_bytes_text = pdf.read()

    with open(path_img, 'rb') as pdf:
        input_bytes_image = pdf.read()

    with open(path_img_text, 'rb') as pdf:
        input_bytes_images_text = pdf.read()

    return {
        'img_text': [(path_img_text, None), (None, input_bytes_images_text)],
        'text': [(path_text, None), (None, input_bytes_text)],
        'img': [(path_img, None), (None, input_bytes_image)],
    }


@pytest.fixture()
def doc_generator_img_text(input_pdf):
    doc_arrays = []
    for uri, buffer in input_pdf['img_text']:
        if uri:
            docs = DocumentArray([Document(uri=uri, mime_type='application/pdf')])
        else:
            docs = DocumentArray([Document(blob=buffer, mime_type='application/pdf')])
        doc_arrays.append(docs)
    return doc_arrays


@pytest.fixture()
def doc_generator_text(input_pdf):
    # import epdb; epdb.serve()
    doc_arrays = []
    for uri, buffer in input_pdf['text']:
        if uri:
            docs = DocumentArray([Document(uri=uri, mime_type='application/pdf')])
        else:
            docs = DocumentArray([Document(blob=buffer, mime_type='application/pdf')])
        doc_arrays.append(docs)
    return doc_arrays


@pytest.fixture()
def doc_generator_img(input_pdf):
    doc_array = []
    for uri, buffer in input_pdf['img']:
        if uri:
            doc = DocumentArray([Document(uri=uri, mime_type='application/pdf')])
        else:
            doc = DocumentArray([Document(blob=buffer, mime_type='application/pdf')])
        doc_array.append(doc)
    return doc_array
