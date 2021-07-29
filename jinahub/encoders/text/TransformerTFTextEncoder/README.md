# TransformerTFTextEncoder
TransformerTFEncoder wraps the tensorflow-version of transformers from huggingface, encodes data from an array of string in size `B` into an ndarray in size `B x D`

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
DETAILSSTART
