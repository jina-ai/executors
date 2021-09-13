# LaserEncoder

**LaserEncoder** is a text encoder based on Facebook Research's LASER encoder.

This encoder is suitable for producing multi-lingual sentence embeddings, enabling
you to have sentences from multiple languages in the same latent space.

### Inputs 

`Document` with `text` to be encoded.

### Returns

`Document` with the `embedding` field filled with an `ndarray` of `dtype=nfloat32`.

## Prerequisites

You should consider downloading the embeddings before starting the executor, and passing
`download_data=False` on initialization. This way your executor won't need to download
the embeddings when it starts up, so it will become available faster. You can download
the embedings like this

```
pip install laserembeddings
python -m laserembeddings download-models
```

## Usage
Use the prebuilt images from JinaHub in your Python code. The input language can be configured with `language`. The full list of possible values can be found at [LASER](https://github.com/facebookresearch/LASER#supported-languages) with the language code ([ISO 639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)) 


Here is an example usage of the **LaserEncoder**.

```python
from jina import Flow, Document

f = Flow().add(uses='jinahub+docker://LaserEncoder')
with f:
    resp = f.post(on='foo', inputs=Document(text='hello Jina'), on_done=print)
```
