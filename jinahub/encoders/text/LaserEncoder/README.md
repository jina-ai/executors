# LaserEncoder

**LaserEncoder** is a encoder based on Facebook Research's LASER (Language-Agnostic SEntence Representations) to compute multilingual sentence embeddings.

It encodes `Document` content from an 1d array of string in size `B` into an ndarray in size `B x D`.




## Prerequisites

> These are only needed if you download the source code and directly use the class. Not needed if you use the Jina Hub method below.

```bash
python -m laserembeddings download-models
```



## Usages

### Via JinaHub

#### using docker images
Use the prebuilt images from JinaHub in your Python code. The input language can be configured with `language`. The full list of possible values can be found at [LASER](https://github.com/facebookresearch/LASER#supported-languages) with the language code ([ISO 639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)) 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://laser-encoder', uses_with={'language': 'en'})
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://laser-encoder'
```

#### using source code
Use the source code from JinaHub in your Python code:

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://laser-encoder')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://laser-encoder'
```


## Example

Here is an example usage of the **LaserEncoder**.

```python
from jina import Flow, Document
f = Flow().add(uses='jinahub+docker://LaserEncoder')
with f:
    resp = f.post(on='foo', inputs=Document(text='hello Jina'), return_results=True)
```

### Inputs 

`Document` with `text` to be encoded.

### Returns

`Document` with `embedding` fields filled with an `ndarray`  with `dtype=nfloat32`.


## Reference
