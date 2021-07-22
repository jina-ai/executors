import shutil

import pytest
import os
import numpy as np
import PIL.Image as Image

from jina import DocumentArray, Document

from jinahub.image.encoder.big_transfer import BigTransferEncoder

directory = os.path.dirname(os.path.realpath(__file__))


def test_initialization_and_model_download():
    shutil.rmtree('pretrained', ignore_errors=True)
    # This call will download the model
    encoder = BigTransferEncoder()
    assert encoder.model_path == 'pretrained'
    assert encoder.model_name == 'R50x1'
    assert not encoder.on_gpu
    assert os.path.exists('pretrained')
    assert os.path.exists(os.path.join('pretrained', 'saved_model.pb'))
    # This call will use the downloaded model
    _ = BigTransferEncoder()
    shutil.rmtree('pretrained', ignore_errors=True)
    with pytest.raises(AttributeError):
        _ = BigTransferEncoder(model_name='model_not_exists')


def test_encoding():
    doc = Document(uri=os.path.join(directory, '../data/test_image.png'))
    doc.convert_image_uri_to_blob()
    img = Image.fromarray(doc.blob.astype('uint8'))
    img = img.resize((96, 96))
    img = np.array(img).astype('float32') / 255
    doc.blob = img
    assert doc.embedding is None

    encoder = BigTransferEncoder()

    encoder.encode(DocumentArray([doc]), {})
    assert doc.embedding.shape == (2048,)


def test_preprocessing():
    doc = Document(uri=os.path.join(directory, '../data/test_image.png'))
    doc.convert_image_uri_to_blob()
    img = Image.fromarray(doc.blob.astype('uint8'))
    img = img.resize((96, 96))
    img = np.array(img).astype('float32') / 255
    doc.blob = img
    assert doc.embedding is None

    encoder = BigTransferEncoder(target_dim=(256, 256, 3))

    encoder.encode(DocumentArray([doc]), {})
    assert doc.embedding.shape == (2048,)


def test_encoding_default_chunks():
    doc = Document(text="testing")
    chunk = Document(uri=os.path.join(directory, '../data/test_image.png'))
    for i in range(3):
        doc.chunks.append(chunk)
        doc.chunks[i].convert_image_uri_to_blob()
        img = Image.fromarray(doc.chunks[i].blob.astype('uint8'))
        img = img.resize((96, 96))
        img = np.array(img).astype('float32') / 255
        doc.chunks[i].blob = img

    encoder = BigTransferEncoder(default_traversal_paths=['c'])

    encoder.encode(DocumentArray([doc]), {})
    assert doc.embedding is None
    for i in range(3):
        assert doc.chunks[i].embedding.shape == (2048,)


def test_encoding_override_chunks():
    doc = Document(text="testing")
    chunk = Document(uri=os.path.join(directory, '../data/test_image.png'))
    for i in range(3):
        doc.chunks.append(chunk)
        doc.chunks[i].convert_image_uri_to_blob()
        img = Image.fromarray(doc.chunks[i].blob.astype('uint8'))
        img = img.resize((96, 96))
        img = np.array(img).astype('float32') / 255
        doc.chunks[i].blob = img

    encoder = BigTransferEncoder()
    assert encoder.default_traversal_paths == ['r']

    encoder.encode(DocumentArray([doc]),
                   parameters={'traversal_paths': ['c']})
    assert doc.embedding is None
    for i in range(3):
        assert doc.chunks[i].embedding.shape == (2048,)
