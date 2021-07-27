# ✨ Image Normalizer

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


<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [🌱 Prerequisites](#-prerequisites)
- [🚀 Usages](#-usages)
- [🎉️ Example](#%EF%B8%8F-example)
- [🔍️ Reference](#%EF%B8%8F-reference)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## 🌱 Prerequisites

No prerequisites are required to run this executor. 

## 🚀 Usages

### 🚚 Via JinaHub

#### using docker images
Use the prebuilt images from JinaHub in your python codes, 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://ImageNormalizer')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://ImageNormalizer'
    override_with: 
      target_size: 42
``` 

#### using source codes
Use the source codes from JinaHub in your python codes,

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://ImageNormalizer')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://ImageNormalizer'
```


### 📦️ Via Pypi

1. Install the `executor-image-normalizer` package.

	```bash
	pip install git+https://github.com/jina-ai/executor-image-normalizer.git
	```

1. Use `executor-image-normalizer` in your code

	```python
	from jina import Flow
	from jinahub.image.normalizer import ImageNormalizer
	
	f = Flow().add(uses=ImageNormalizer)
	```


### 🐳 Via Docker

1. Clone the repo and build the docker image

	```shell
	git clone https://github.com/jina-ai/executor-image-normalizer.git
	cd executor-image-normalizer
	docker build -t image-normalizer .
	```

1. Use `image-normalizer` in your codes

	```python
	from jina import Flow
	
	f = Flow().add(uses='docker://image-normalizer')
	```
	

## 🎉️ Example 


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


## 🔍️ Reference
