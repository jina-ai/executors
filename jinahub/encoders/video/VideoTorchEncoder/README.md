# ✨ VideoTorchEncoder

**VideoTorchEncoder** is a class that encodes video clips into dense embeddings using pretrained models 
from [`torchvision.models`](https://pytorch.org/docs/stable/torchvision/models.html) for video data.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [🌱 Prerequisites](#-prerequisites)
- [🚀 Usages](#-usages)
- [🎉️ Example](#%EF%B8%8F-example)
- [🔍️ Reference](#%EF%B8%8F-reference)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## 🌱 Prerequisites

To install the dependencies locally run 
```
pip install . 
pip install -r tests/requirements.txt
```
To verify the installation works:
```
pytest tests
```

## 🚀 Usages

### 🚚 Via JinaHub

#### using docker images
Use the prebuilt images from JinaHub in your python codes, 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://VideoTorchEncoder',
               volumes='/your_home_folder/.cache/torch:/root/.cache/torch')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://VideoTorchEncoder'
    volumes: '/your_home_folder/.cache/torch:/root/.cache/torch'
```

#### using source codes
Use the source codes from JinaHub in your python codes,

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://VideoTorchEncoder')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://VideoTorchEncoder'
```


### 📦️ Via Pypi

1. Install the `jinahub-video-torch-encoder` package.

	```bash
	pip install git+https://github.com/jina-ai/EXECUTOR_REPO_NAME.git
	```

2. Use `jinahub-video-torch-encoder` in your code

	```python
	from jina import Flow
	from from jinahub.encoder.video_torch_encoder import VideoTorchEncoder
	
	f = Flow().add(uses=VideoTorchEncoder)
	```


### 🐳 Via Docker

1. Clone the repo and build the docker image

	```shell
	git clone https://github.com/jina-ai/executor-video-torch-encoder.git
	cd executor-video-torch-encoder
	docker build -t video-torch-encoder-image .
	```

2. Use `video-torch-encoder` in your codes

	```python
	from jina import Flow
	
	f = Flow().add(uses='docker://video-torch-encoder-image:latest',
                   volumes='/your_home_folder/.cache/torch:/root/.cache/torch')
	```
	

## 🎉️ Example 


```python
from jina import Flow, Document
from torchvision.io.video import read_video

f = Flow().add(uses='jinahub+docker://VideoTorchEncoder')

video_array, _, _ = read_video('your_video.mp4')  # video frames in the shape of `NumFrames x Height x Width x 3`

video_array = video_array.cpu().detach().numpy()

with f:
    resp = f.post(on='foo', inputs=[Document(blob=video_array), ], return_results=True)
    assert resp[0].docs[0].embedding.shape == (512,)
```

### Inputs 

`Documents` must have `blob` of the shape `Channels x NumFrames x 112 x 112`, if `use_default_preprocessing=False`.
When setting `use_default_preprocessing=True`, the input `blob` must have the size of `Frame x Height x Width x Channel`.

### Returns

`Document` with `embedding` fields filled with an `ndarray` of the shape `512`.


## 🔍️ Reference
- https://pytorch.org/vision/stable/models.html#video-classification

