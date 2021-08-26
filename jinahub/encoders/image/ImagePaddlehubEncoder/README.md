# ImagePaddlehubEncoder 

**ImagePaddlehubEncoder** encodes `Document` content from a ndarray, potentially B x (Channel x Height x Width) into a ndarray of `B x D`. Internally, **ImagePaddlehubEncoder** wraps the models from [paddlehub](https://github.com/PaddlePaddle/PaddleHub)

**Table of Contents**

- [ImagePaddlehubEncoder](#imagepaddlehubencoder)
  - [🌱 Prerequisites](#-prerequisites)
  - [🚀 Usages](#-usages)
    - [🚚 Via JinaHub](#-via-jinahub)
      - [using docker images](#using-docker-images)
      - [using source code](#using-source-code)
  - [🎉 Example:](#-example)
    - [Inputs](#inputs)
    - [Returns](#returns)
  - [🔍️ Reference](#️-reference)


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
	
f = Flow().add(uses='jinahub+docker://ImagePaddlehubEncoder')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://ImagePaddlehubEncoder'
```

#### using source code
Use the source code from JinaHub in your Python code:

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://ImagePaddlehubEncoder')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://ImagePaddlehubEncoder'
```
## 🎉 Example:

Here is an example usage of the **ImagePaddlehubEncoder**.

```python
    def process_response(resp):
        ...
    f = Flow().add(uses={
        'jtype': ImagePaddlehubEncoder.__name__,
        'with': {
            'default_batch_size': 32,
            'model_name': 'xception71_imagenet',
        },
        'metas': {
            'py_modules': ['paddle_image.py']
        }
    })
    with f:
        f.post(on='/test', inputs=(Document(blob=np.ones((224, 224, 3))) for _ in range(25)), on_done=process_response)
```

### Inputs 

`Document` with `blob` as data of images.

### Returns

`Document` with `embedding` fields filled with an `ndarray`  with `dtype=nfloat32`.

## 🔍️ Reference
