__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os

import pytest
import torch
import numpy as np

import torchvision.models.video as models
from torchvision import transforms

from jina import Document, DocumentArray

try:
    from video_torch_encoder import VideoTorchEncoder, ConvertFHWCtoFCHW, ConvertFCHWtoCFHW
except:
    from jinahub.encoder.video_torch_encoder import VideoTorchEncoder, ConvertFHWCtoFCHW, ConvertFCHWtoCFHW

cur_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize('model_name', ['r3d_18', 'mc3_18', 'r2plus1d_18'])
def test_video_torch_encoder(model_name):
    ex = VideoTorchEncoder(model_name=model_name, use_default_preprocessing=False)
    da = DocumentArray([Document(blob=np.random.random((3, 2, 224, 224))) for _ in range(10)])
    ex.encode(da, {})
    assert len(da) == 10
    for doc in da:
        assert doc.embedding.shape == (512,)


@pytest.mark.parametrize('batch_size', [1, 3, 10])
def test_video_torch_encoder_traversal_paths(batch_size):
    ex = VideoTorchEncoder(use_default_preprocessing=False)

    def _create_doc_with_video_chunks():
        d = Document(blob=np.random.random((3, 2, 112, 112)))
        d.chunks = [Document(blob=np.random.random((3, 2, 112, 112))) for _ in range(5)]
        return d

    da = DocumentArray([_create_doc_with_video_chunks() for _ in range(10)])
    ex.encode(da, {'traversal_paths': ['r', 'c'], 'batch_size': batch_size})
    assert len(da) == 10
    for doc in da:
        assert doc.embedding.shape == (512,)
        assert len(doc.chunks) == 5
        for chunk in doc.chunks:
            assert chunk.embedding.shape == (512,)


@pytest.mark.parametrize('model_name', ['r3d_18', 'mc3_18', 'r2plus1d_18'])
def test_video_torch_encoder_use_default_preprocessing(model_name):
    ex = VideoTorchEncoder(model_name=model_name, use_default_preprocessing=True)
    da = DocumentArray([Document(blob=np.random.random((10, 270, 480, 3))) for _ in range(10)])
    ex.encode(da, {})
    assert len(da) == 10
    for doc in da:
        assert doc.embedding.shape == (512,)


@pytest.fixture()
def kinects_videos():
    from torchvision.datasets import Kinetics400

    dataset = Kinetics400(root=os.path.join(cur_dir, '../data/kinetics400'), frames_per_clip=20)
    return [dataset[0][0], dataset[0][0]]


@pytest.mark.parametrize('model_name', ['mc3_18', 'r2plus1d_18', 'r3d_18'])
def test_with_dataset_video(model_name, kinects_videos):
    da = DocumentArray([Document(blob=video.detach().numpy()) for video in kinects_videos])

    ex = VideoTorchEncoder(use_default_preprocessing=True, model_name=model_name)
    ex.encode(da, {})
    assert len(da) == 2
    for doc in da:
        assert doc.embedding.shape == (512,)

    model = getattr(models, model_name)(pretrained=True).eval()
    mean = (0.43216, 0.394666, 0.37645)
    std = (0.22803, 0.22145, 0.216989)
    resize_size = (128, 171)
    crop_size = (112, 112)
    t = transforms.Compose([
        ConvertFHWCtoFCHW(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Resize(resize_size),
        transforms.Normalize(mean=mean, std=std),
        transforms.CenterCrop(crop_size),
        ConvertFCHWtoCFHW()
    ])
    tensor = torch.stack([t(video) for video in kinects_videos])

    def _get_embeddings(x) -> torch.Tensor:
        embeddings = torch.Tensor()

        def get_activation(model, model_input, output):
            nonlocal embeddings
            embeddings = output

        handle = model.avgpool.register_forward_hook(get_activation)
        model(x)
        handle.remove()
        return embeddings.flatten(1)

    embedding_batch = _get_embeddings(tensor)
    for doc, expected_torch_embedding in zip(da, embedding_batch):
        np.testing.assert_almost_equal(doc.embedding, expected_torch_embedding.detach().numpy())
