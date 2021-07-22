# ImagePaddlehubEncoder

**ImagePaddlehubEncoder** encodes `Document` content from a ndarray, potentially B x (Channel x Height x Width) into a ndarray of `B x D`. Internally, **ImagePaddlehubEncoder** wraps the models from [paddlehub](https://github.com/PaddlePaddle/PaddleHub)

## 🚀 Usages

To install the dependencies locally run 
```
pip install . 
pip install -r tests/requirements.txt
```
To verify the installation works:
```
pytest tests
```

### 🚚 Via JinaHub

#### using docker images
Use the prebuilt images from JinaHub in your python codes, 

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

#### using source codes
Use the source codes from JinaHub in your python codes,

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


### 📦️ Via Pypi

1. Install the package.

	```bash
	pip install git+https://github.com/jina-ai//executor-image-paddle-encoder.git
	```

1. Use `ImagePaddlehubEncoder` in your code

	```python
	from jina import Flow
	from jinahub.encoder.paddle_image import ImagePaddlehubEncoder
	
	f = Flow().add(uses=ImagePaddlehubEncoder)
	```


### 🐳 Via Docker

1. Clone the repo and build the docker image

	```shell
	git clone https://github.com/jina-ai/executor-image-paddle-encoder.git
	cd executor-image-paddle-encoder
	docker build -t executor-image-paddle-encoder .
	```

1. Use `executor-image-paddle-encoder` in your codes

	```python
	from jina import Flow
	
	f = Flow().add(uses='docker://executor-image-paddle-encoder:latest')
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
