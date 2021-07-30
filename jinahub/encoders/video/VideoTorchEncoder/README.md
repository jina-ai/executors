# ✨ VideoTorchEncoder

**VideoTorchEncoder** is a class that encodes video clips into dense embeddings using pretrained models 
from [`torchvision.models`](https://pytorch.org/docs/stable/torchvision/models.html) for video data.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [🌱 Prerequisites](#-prerequisites)
- [🚀 Usages](#-usages)
- [🎉️ Example](#-example)
- [🔍️ Reference](#-reference)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## 🌱 Prerequisites


> These are only needed if you download the source code and directly use the class. Not needed if you use the Jina Hub method below.

To install the dependencies locally, run 
```
pip install -r requirements.txt
```

## 🚀 Usages

### 🚚 Via JinaHub

#### using docker images
Use the prebuilt images from JinaHub in your Python code: 

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

#### using source code
Use the source code from JinaHub in your Python code:

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

