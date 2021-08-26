# TransformerTFTextEncoder
TransformerTFEncoder wraps the tensorflow-version of transformers from huggingface, encodes data from an array of string in size `B` into an ndarray in size `B x D`

**Table of Contents**

- [🌱 Prerequisites](#-prerequisites)
- [🚀 Usages](#-usages)
- [🎉️ Example](#-example)
- [🔍️ Reference](#-reference)

## 🌱 Prerequisites

> These are only needed if you download the source code and directly use the class. Not needed if you use the Jina Hub method below.

In case you want to install the dependencies locally run 
```
pip install -r requirements.txt
```

## 🚀 Usages

### 🚚 Via JinaHub

#### using docker images
Use the prebuilt images from JinaHub in your Python code: 

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

#### using source code
Use the source code from JinaHub in your Python code:

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

## 🔍️ Reference
