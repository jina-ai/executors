import torch
import os
import clip
import numpy as np
from glob import glob
from PIL import Image

from jina import Flow, Document
from jinahub.encoder.clip_image import CLIPImageEncoder

cur_dir = os.path.dirname(os.path.abspath(__file__))

def test_clip_data():
    docs = []
    for file in glob(os.path.join(cur_dir, 'data', '*')):
        pil_image = Image.open(file)
        nd_image = np.array(pil_image)
        docs.append(Document(id=file, blob=nd_image))

    with Flow().add(uses=CLIPImageEncoder) as f:
        results = f.post(on='/test', inputs=docs, return_results=True)
        os.path.join(cur_dir, 'data', 'banana2.png')
        image_name_to_ndarray = {}
        for d in results[0].docs:
            image_name_to_ndarray[d.id] = d.embedding

    def dist(a, b):
        nonlocal image_name_to_ndarray
        a_embedding = image_name_to_ndarray[os.path.join(cur_dir, 'data', f'{a}.png')]
        b_embedding = image_name_to_ndarray[os.path.join(cur_dir, 'data', f'{b}.png')]
        return np.linalg.norm(a_embedding - b_embedding)

    # assert semantic meaning is captured in the encoding
    small_distance = dist('banana1', 'banana2')
    assert small_distance < dist('banana1', 'airplane')
    assert small_distance < dist('banana1', 'satellite')
    assert small_distance < dist('banana1', 'studio')
    assert small_distance < dist('banana2', 'airplane')
    assert small_distance < dist('banana2', 'satellite')
    assert small_distance < dist('banana2', 'studio')
    assert small_distance < dist('airplane', 'studio')
    assert small_distance < dist('airplane', 'satellite')
    assert small_distance < dist('studio', 'satellite')

    # assert same results like calculating it manually
    model, preprocess = clip.load('ViT-B/32', device='cpu')
    assert len(image_name_to_ndarray) == 5
    for file, actual_embedding in image_name_to_ndarray.items():
        image = preprocess(Image.open(file)).unsqueeze(0).to('cpu')

        with torch.no_grad():
            expected_embedding = model.encode_image(image).numpy()[0]

        np.testing.assert_almost_equal(actual_embedding, expected_embedding, 5)
