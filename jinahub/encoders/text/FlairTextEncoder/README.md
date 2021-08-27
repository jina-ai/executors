# FlairTextEncoder

`FlairTextEncoder` encodes `Document` content from an array of string in size `B` into a ndarray in size `B x D`.
 
Internally, `FlairTextEncoder` wraps the DocumentPoolEmbeddings from Flair.




## Usages

### Via JinaHub

#### using docker images
Use the prebuilt images from JinaHub in your Python code: 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://FlairTextEncoder')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://FlairTextEncoder'
```

#### using source code
Use the source code from JinaHub in your Python code:

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://FlairTextEncoder')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://FlairTextEncoder'
```


## Example

Here is an example usage of the **FlairTextEncoder**.

```python
from jina import Flow, Document
f = Flow().add(uses='jinahub+docker://FlairTextEncoder')
with f:
    resp = f.post(on='foo', inputs=Document(text='hello Jina'), return_results=True)
```

### Inputs 

`Document` with `text` to be encoded.

### Returns

`Document` with `embedding` fields filled with an `ndarray`  with `dtype=nfloat32`.


