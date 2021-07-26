# TransformerTFTextEncoder
TransformerTFEncoder wraps the tensorflow-version of transformers from huggingface, encodes data from an array of string in size `B` into an ndarray in size `B x D`

## 🚀 Usages

### 🚚 Via JinaHub

#### using docker images
Use the prebuilt images from JinaHub in your python codes, 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://TransformerTFTextEncoder')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://TransformerTFTextEncoder'
```

#### using source codes
Use the source codes from JinaHub in your python codes,

```python
from jina import Flow, Document

f = Flow().add(uses='jinahub://TransformerTFTextEncoder')

with f:
    resp = f.post(on='foo', inputs=Document(text='hello Jina'), return_results=True)
    print(f'{resp[0].docs[0].embedding.shape}')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://TransformerTFTextEncoder'
```


### 📦️ Via Pypi

1. Install the package.

	```bash
	pip install git+https://github.com/jina-ai/executor-text-transformer-tf-encoder.git
	```

1. Use `TransformerTFTextEncoder` in your code

	```python
	from jina import Flow
	from jinahub.encoder.transformer_tf_text_encode import TransformerTFTextEncoder
	
	f = Flow().add(uses=TransformerTFTextEncoder)
	```


### 🐳 Via Docker

1. Clone the repo and build the docker image

	```shell
	git clone https://github.com/jina-ai/executor-text-transformer-tf-encoder.git
	cd executor-text-transformer-tf-encoder
	docker build -t executor-text-transformer-tf-encoder .
	```

1. Use `executor-text-transformer-tf-encoder` in your codes

	```python
	from jina import Flow
	
	f = Flow().add(uses='docker://executor-text-transformer-tf-encoder:latest')
	```
 
## 🎉 Example:

Here is an example usage of the **TransformerTFTextEncoder**.

```python
    def process_response(resp):
        print(resp)

    f = Flow().add(uses={
        'jtype': TransformerTFTextEncoder.__name__,
        'with': {
            'pretrained_model_name_or_path': 'distilbert-base-uncased'
        },
        'metas': {
            'py_modules': ['transformer_tf_text_encode.py']
        }
    })
    with f:
        f.post(on='/test', inputs=(Document(text='hello Jina', on_done=process_response)))
```

### Inputs 

`Document` with `blob` as data of text.

### Returns

`Document` with `embedding` fields filled with an `ndarray`  with `dtype==np.float32`.
