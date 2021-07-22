# ✨ Transformer Torch Encoder

**Transformer Torch Encoder** is a class that encodes sentences into embeddings.

The following parameters can be used:

- `pretrained_model_name_or_path` (str, default 'sentence-transformers/distilbert-base-nli-stsb-mean-tokens'): Path to pretrained model or name of the model in transformers package
- `base_tokenizer_model` (str, optional): Base tokenizer model
- `pooling_strategy` (str, default 'mean'): Pooling Strategy
- `layer_index` (int, default -1): Index of the layer which contains the embeddings
- `max_length` (int, optional): Max length argument for the tokenizer
- `embedding_fn_name` (str, default __call__): Function to call on the model in order to get output
- `device` (str, default 'cpu'): Device to be used. Use 'cuda' for GPU
- `default_traversal_paths` (List[str], Optional): Used in the encode method an define traversal on the received `DocumentArray`. Defaults to ['r']
- `default_batch_size` (int, default 32): Defines the batch size for inference on the loaded PyTorch model.


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
	
f = Flow().add(uses='jinahub+docker://TransformerTorchEncoder',
               volumes='/your_user/.cache/huggingface:/root/.cache/huggingface')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://TransformerTorchEncoder'
    volumes: '/your_user/.cache/huggingface:/root/.cache/huggingface'
    overwrite_with: 
      target_size: 42
``` 
This does not support GPU at the moment.

#### using source codes
Use the source codes from JinaHub in your python codes,

```python
from jina import Flow, Document

f = Flow().add(uses='jinahub://TransformerTorchEncoder')

with f:
    resp = f.post(on='foo', inputs=Document(text='hello Jina'), return_results=True)
    print(f'{resp[0].docs[0].embedding.shape}')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://TransformerTorchEncoder'
```


### 📦️ Via Pypi

1. Install the `executor-transformer-torch-encoder` package.

	```bash
	pip install git+https://github.com/jina-ai/executor-transformer-torch-encoder.git
	```

1. Use `executor-transformer-torch-encoder` in your code

	```python
	from jina import Flow
	from jinahub.text.encoders.transform_encoder import TransformerTorchEncoder

	
	f = Flow().add(uses=TransformerTorchEncoder)
	```


### 🐳 Via Docker

1. Clone the repo and build the docker image

	```shell
	git clone https://github.com/jina-ai/executor-transformer-torch-encoder.git
	cd executor-transformer-torch-encoder
	docker build -t transformer-torch-encoder .
	```
    Alternatively, build the GPU docker image:
    ```shell
    docker build -f Dockerfile.gpu -t transformer-torch-encoder .
	```

1. Use `transformer-torch-encoder` in your codes

	```python
	from jina import Flow
	
	f = Flow().add(uses='docker://transformer-torch-encoder')
	```
    Or, when using the GPU image:
	```python
	from jina import Flow
	
	f = Flow().add(uses='docker://transformer-torch-encoder', docker_kwargs={'runtime': 'nvidia'})
	```
	

## 🎉️ Example 


```python
from jina import Flow, Document

f = Flow().add(uses='jinahub+docker://TransformerTorchEncoder')

with f:
    resp = f.post(on='foo', inputs=Document(), return_results=True)
	print(f'{resp}')
```

### Inputs 

`Document` with text in the `text` field.

### Returns

`Document` with `embedding` fields filled with an `ndarray` of the shape `embedding_dim` (model-dependent) with `dtype=nfloat32`.


## 🔍️ Reference
- Available models: https://huggingface.co/transformers/pretrained_models.html
- More available models: https://huggingface.co/sentence-transformers
