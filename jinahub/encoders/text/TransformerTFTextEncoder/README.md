# TransformerTFTextEncoder
TransformerTFEncoder wraps the tensorflow-version of transformers from huggingface, encodes data from an array of string in size `B` into an ndarray in size `B x D`




## Usage

```python
from jina import Flow, Document

f = Flow().add(uses='jinahub://TransformerTFTextEncoder')

with f:
    resp = f.post(on='foo', inputs=Document(text='hello Jina'), return_results=True)
    print(f'{resp[0].docs[0].embedding.shape}')
```


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


