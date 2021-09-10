# VideoTorchEncoder

**VideoTorchEncoder** is a class that encodes video clips into dense embeddings using pretrained models 
from [`torchvision.models`](https://pytorch.org/docs/stable/torchvision/models.html) for video data.





## Usage 

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


## Reference
- https://pytorch.org/vision/stable/models.html#video-classification

