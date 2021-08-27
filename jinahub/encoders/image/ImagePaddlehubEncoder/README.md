# ImagePaddlehubEncoder

**ImagePaddlehubEncoder** encodes `Document` content from a ndarray, potentially B x (Channel x Height x Width) into a ndarray of `B x D`. Internally, **ImagePaddlehubEncoder** wraps the models from [paddlehub](https://github.com/PaddlePaddle/PaddleHub)


## Usage

Here is an example usage of the **ImagePaddlehubEncoder**.

```python
    def process_response(resp):
        ...
    f = Flow().add(uses={
        'jtype': ImagePaddlehubEncoder.__name__,
        'with': {
            'default_batch_size': 32,
            'model_name': 'xception71_imagenet',
        },
        'metas': {
            'py_modules': ['paddle_image.py']
        }
    })
    with f:
        f.post(on='/test', inputs=(Document(blob=np.ones((224, 224, 3))) for _ in range(25)), on_done=process_response)
```

### Inputs 

`Document` with `blob` as data of images.

### Returns

`Document` with `embedding` fields filled with an `ndarray`  with `dtype=nfloat32`.

