# LaserEncoder

**LaserEncoder** is a encoder based on Facebook Research's LASER (Language-Agnostic SEntence Representations) to compute multilingual sentence embeddings.

It encodes `Document` content from an 1d array of string in size `B` into an ndarray in size `B x D`.




## Prerequisites

> These are only needed if you use `jinahub://LaserEncoder`.

```bash
python -m laserembeddings download-models
```


## Reference
- [LASER](https://github.com/facebookresearch/LASER#supported-languages) with the language code ([ISO 639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)) 

