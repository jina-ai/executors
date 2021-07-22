import os
import shutil
import pytest

import PIL.Image as Image
import numpy as np

from jina import Flow, Document

cur_dir = os.path.dirname(os.path.abspath(__file__))


def data_generator(num_docs):
    for i in range(num_docs):
        doc = Document(
            uri=os.path.join(cur_dir, '..', 'data', 'test_image.png'))
        doc.convert_image_uri_to_blob()
        img = Image.fromarray(doc.blob.astype('uint8'))
        img = img.resize((96, 96))
        img = np.array(img).astype('float32') / 255
        doc.blob = img
        yield doc


@pytest.mark.parametrize(
    'model_name', ['R50x1', 'R101x1', 'R50x3', 'R101x3']  #, 'R152x4']
)
def test_all_models(model_name: str):
    shutil.rmtree('pretrained', ignore_errors=True)
    os.environ['TRANSFER_MODEL_NAME'] = model_name
    with Flow.load_config(os.path.join(cur_dir, 'flow.yml')) as flow:
        data = flow.post(on='/index', inputs=data_generator(100),
                         request_size=10, return_results=True)
        docs = data[0].docs
        for doc in docs:
            assert doc.embedding is not None
