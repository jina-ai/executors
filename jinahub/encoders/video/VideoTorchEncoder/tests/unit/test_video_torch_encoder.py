__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision.models.video as models
from jina import Document, DocumentArray, Executor
from torchvision import transforms
from video_torch_encoder import ConvertFCHWtoCFHW, ConvertFHWCtoFCHW, VideoTorchEncoder


@pytest.fixture(scope="module")
def encoder() -> VideoTorchEncoder:
    return VideoTorchEncoder()


@pytest.fixture(scope="module")
def encoder_with_processing() -> VideoTorchEncoder:
    return VideoTorchEncoder(use_preprocessing=True)


@pytest.fixture()
def kinects_videos():
    from torchvision.datasets import Kinetics400

    dataset = Kinetics400(
        root=Path(__file__).parents[1] / 'data/kinetics400', frames_per_clip=20
    )
    return [dataset[0][0], dataset[0][0]]


@pytest.fixture()
def random_doc_cnhw():
    """Random document of (channel, num_frame, height, width)"""
    return Document(blob=np.random.random((3, 2, 224, 224)))


@pytest.fixture()
def random_doc_nhwc():
    """Random document of (num_frame, height, width, channel) allow pre-processing."""
    return Document(blob=np.random.random((2, 224, 224, 3)))


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[2] / 'config.yml'))
    assert ex.batch_size == 32


def test_no_documents(encoder: VideoTorchEncoder):
    docs = DocumentArray()
    encoder.encode(docs=docs, parameters={})
    assert len(docs) == 0  # SUCCESS


def test_none_docs(encoder: VideoTorchEncoder):
    encoder.encode(docs=None, parameters={})


def test_docs_no_blobs(encoder: VideoTorchEncoder):
    docs = DocumentArray([Document()])
    encoder.encode(docs=DocumentArray(), parameters={})
    assert len(docs) == 1
    assert docs[0].embedding is None


@pytest.mark.parametrize('use_preprocessing', [True, False])
@pytest.mark.parametrize('model_name', ['r3d_18', 'mc3_18', 'r2plus1d_18'])
def test_encode_single_document(
    model_name, use_preprocessing, random_doc_cnhw, random_doc_nhwc
):
    ex = VideoTorchEncoder(
        model_name=model_name,
        use_preprocessing=use_preprocessing,
        download_progress=False,
    )
    da = DocumentArray()
    if use_preprocessing:
        da.append(random_doc_nhwc)
    else:
        da.append(random_doc_cnhw)
    ex.encode(da, {})
    assert len(da) == 1
    for doc in da:
        assert doc.embedding.shape == (512,)


@pytest.mark.parametrize('use_preprocessing', [True, False])
@pytest.mark.parametrize('model_name', ['r3d_18', 'mc3_18', 'r2plus1d_18'])
def test_encode_single_document_given_wrong_input_shape(
    model_name, use_preprocessing, random_doc_cnhw, random_doc_nhwc
):
    ex = VideoTorchEncoder(
        model_name=model_name,
        use_preprocessing=use_preprocessing,
        download_progress=False,
    )
    da = DocumentArray()
    if use_preprocessing:
        da.append(random_doc_cnhw)
    else:
        da.append(random_doc_nhwc)
    with pytest.raises(RuntimeError):
        ex.encode(da, {})


@pytest.mark.parametrize('use_preprocessing', [True, False])
@pytest.mark.parametrize('model_name', ['r3d_18', 'mc3_18', 'r2plus1d_18'])
def test_encode_multiple_documents(
    model_name, use_preprocessing, random_doc_cnhw, random_doc_nhwc
):
    ex = VideoTorchEncoder(
        model_name=model_name,
        use_preprocessing=use_preprocessing,
        download_progress=False,
    )
    da = DocumentArray()
    if use_preprocessing:
        da.extend([random_doc_nhwc, random_doc_nhwc])
    else:
        da.extend([random_doc_cnhw, random_doc_cnhw])
    ex.encode(da, {})
    assert len(da) == 2
    for doc in da:
        assert doc.embedding.shape == (512,)


@pytest.mark.parametrize('batch_size', [1, 3, 10])
def test_video_torch_encoder_traversal_paths(batch_size):
    ex = VideoTorchEncoder(use_preprocessing=False, download_progress=False)

    def _create_doc_with_video_chunks():
        d = Document(blob=np.random.random((3, 2, 112, 112)))
        d.chunks = [Document(blob=np.random.random((3, 2, 112, 112))) for _ in range(5)]
        return d

    da = DocumentArray([_create_doc_with_video_chunks() for _ in range(10)])
    ex.encode(da, {'traversal_paths': 'r,c', 'batch_size': batch_size})
    assert len(da) == 10
    for doc in da:
        assert doc.embedding.shape == (512,)
        assert len(doc.chunks) == 5
        for chunk in doc.chunks:
            assert chunk.embedding.shape == (512,)


@pytest.mark.parametrize('model_name', ['mc3_18', 'r2plus1d_18', 'r3d_18'])
def test_with_dataset_video(model_name, kinects_videos):
    da = DocumentArray(
        [Document(blob=video.detach().numpy()) for video in kinects_videos]
    )

    ex = VideoTorchEncoder(
        use_preprocessing=True,
        model_name=model_name,
        download_progress=False,
    )
    ex.encode(da, {})
    assert len(da) == 2
    for doc in da:
        assert doc.embedding.shape == (512,)

    model = getattr(models, model_name)(pretrained=True, progress=False).eval()
    mean = (0.43216, 0.394666, 0.37645)
    std = (0.22803, 0.22145, 0.216989)
    resize_size = (128, 171)
    crop_size = (112, 112)
    t = transforms.Compose(
        [
            ConvertFHWCtoFCHW(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize(resize_size),
            transforms.Normalize(mean=mean, std=std),
            transforms.CenterCrop(crop_size),
            ConvertFCHWtoCFHW(),
        ]
    )
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
        np.testing.assert_almost_equal(
            doc.embedding, expected_torch_embedding.detach().numpy()
        )


@pytest.mark.gpu
@pytest.mark.parametrize('use_preprocessing', [True, False])
@pytest.mark.parametrize('model_name', ['r3d_18', 'mc3_18', 'r2plus1d_18'])
def test_video_torch_encoder_gpu(
    model_name, use_preprocessing, random_doc_nhwc, random_doc_cnhw
):
    ex = VideoTorchEncoder(
        model_name=model_name,
        use_preprocessing=use_preprocessing,
        device='cuda',
        download_progress=False,
    )
    if use_preprocessing:
        da = DocumentArray([random_doc_nhwc for _ in range(10)])
    else:
        da = DocumentArray([random_doc_cnhw for _ in range(10)])
    assert ex.device == 'cuda'
    ex.encode(da, {})
    assert len(da) == 10
    for doc in da:
        assert doc.embedding.shape == (512,)
