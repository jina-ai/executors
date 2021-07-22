__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os

import numpy as np

from jina import Document, Flow, DocumentArray

try:
    from custom_image_torch_encoder import CustomImageTorchEncoder
except:
    from jinahub.encoder.custom_image_torch_encoder import CustomImageTorchEncoder

cur_dir = os.path.dirname(os.path.abspath(__file__))


def test_video_torch_encoder():
    model_state_dict_path = os.path.join(cur_dir, '../model/model_state_dict.pth')
    input_dim = 224
    test_img = np.random.rand(3, input_dim, input_dim)
    docs = DocumentArray([Document(blob=test_img), Document(blob=test_img)])
    f = Flow().add(uses={'jtype': 'CustomImageTorchEncoder',
                         'with': {'model_state_dict_path': model_state_dict_path,
                                  'layer_name': 'conv1',
                                  'model_definition_file': os.path.join(cur_dir, '../model/external_model.py'),
                                  'model_class_name': 'ExternalModel'}})
    with f:
        resp = f.post(on='/test', inputs=docs,
                      return_results=True)
        assert resp[0].docs[0].embedding.shape == (10,)
        assert resp[0].docs[1].embedding.shape == (10,)
