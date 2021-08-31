# Image Normalizer 

**Image Normalizer** is a class that resizes, crops and normalizes images.
Since normalization is highly dependent on the model, 
it is recommended to have it as part of the encoders instead of using it in a dedicated executor.
Therefore, misconfigurations resulting in a training-serving-gap are less likely.

The following parameters can be used:

- `resize_dim` (int): The size of the image after resizing
- `target_size` (tuple or int): The dimensions to crop the image to (center cropping is used)
- `img_mean` (tuple, default (0,0,0)): The mean for normalization
- `img_std` (tuple, default (1,1,1)): The standard deviation for normalization
- `channel_axis` (int): The channel axis in the images used
- `target_channel_axis` (int): The desired channel axis in the images. If this is not equal to the channel_axis, the axis is moved.
- `target_dtype` (np.dtype, default `np.float32`): The desired type of the image array 


## Usage 


```python
from jina import Flow, Document

f = Flow().add(uses='jinahub+docker://ImageNormalizer')

with f:
    resp = f.post(on='foo', inputs=Document(), return_results=True)
    print(f'{resp}')
```

### Inputs 

`Document` with image `blob`.

### Returns

`Document` with overridden image `blob` that is normalized, scaled, cropped and resized as instructed.

