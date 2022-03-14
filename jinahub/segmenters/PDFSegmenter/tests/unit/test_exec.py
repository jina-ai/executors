__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os

import pytest
from jina import Document, DocumentArray, Executor
from pdf_segmenter import PDFSegmenter
from PIL import Image


@pytest.fixture()
def executor():
    return PDFSegmenter()


@pytest.fixture()
def executor_from_config():
    return Executor.load_config('config.yml')


def test_empty_docs(executor):
    da = DocumentArray()
    executor.craft(da)
    assert len(da) == 0


def test_none_input(executor):
    executor.craft(None)


def test_io_images_and_text(
    executor_from_config, test_dir, doc_generator_img_text, expected_text
):
    doc_array = doc_generator_img_text
    assert len(doc_array) > 0
    for doc in doc_array:
        executor_from_config.craft(doc)
        chunks = doc[0].chunks
        assert len(chunks) == 3
        # Check images
        for idx, c in enumerate(chunks[:2]):
            with Image.open(os.path.join(test_dir, f'data/test_img_{idx}.jpg')) as img:
                tensor = chunks[idx].tensor
                assert chunks[idx].mime_type == 'image/*'
                assert tensor.shape[1], tensor.shape[0] == img.size
                if idx == 0:
                    assert tensor.shape == (660, 1024, 3)
                if idx == 1:
                    assert tensor.shape == (626, 1191, 3)

            # Check text
            assert chunks[2].text == expected_text
            assert chunks[2].mime_type == 'text/plain'


def test_io_text(executor_from_config, doc_generator_text, expected_text):
    doc_arrays = doc_generator_text
    assert len(doc_arrays) > 0
    for docs in doc_arrays:
        executor_from_config.craft(docs)
        chunks = docs[0].chunks
        assert len(chunks) == 1
        # Check test
        assert chunks[0].text == expected_text
        assert chunks[0].mime_type == 'text/plain'


def test_io_img(executor_from_config, test_dir, doc_generator_img):
    doc_arrays = doc_generator_img
    assert len(doc_arrays) > 0
    for docs in doc_arrays:
        executor_from_config.craft(docs)
        chunks = docs[0].chunks
        assert len(chunks) == 3
        # Check images
        for idx, c in enumerate(chunks[:2]):
            with Image.open(os.path.join(test_dir, f'data/test_img_{idx}.jpg')) as img:
                tensor = chunks[idx].tensor
                assert chunks[idx].mime_type == 'image/*'
                assert tensor.shape[1], tensor.shape[0] == img.size
                if idx == 0:
                    assert tensor.shape == (660, 1024, 3)
                if idx == 1:
                    assert tensor.shape == (626, 1191, 3)
