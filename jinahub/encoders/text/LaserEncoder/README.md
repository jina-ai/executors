# LaserEncoder

**LaserEncoder** is a text encoder based on Facebook Research's LASER encoder.

This encoder is suitable for producing multi-lingual sentence embeddings, enabling
you to have sentences from multiple languages in the same latent space.

## Prerequisites

> These are only needed if you use `jinahub://LaserEncoder`.

You should consider downloading the embeddings before starting the executor, and passing
`download_data=False` on initialization. This way your executor won't need to download
the embeddings when it starts up, so it will become available faster. You can download
the embedings like this

```
pip install laserembeddings
python -m laserembeddings download-models
```


## Reference
- [LASER](https://github.com/facebookresearch/LASER#supported-languages) with the language code ([ISO 639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes))

<!-- version=v0.2 -->
