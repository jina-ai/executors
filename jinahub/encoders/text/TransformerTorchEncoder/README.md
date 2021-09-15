# TransformerTorchEncoder

**TransformerTorchEncoder** wraps the torch-version of transformers from huggingface. It encodes text data into dense vectors.

**TransformerTorchEncoder** receives [`Documents`](https://docs.jina.ai/fundamentals/document/) with `text` attributes.
The `text` attribute represents the text to be encoded. This Executor will encode each `text` into a dense vector and store them in the `embedding` attribute of the `Document`.


## Usage


Use the prebuilt images from Jina Hub in your Flow and encode an image:

```python
from jina import Flow, Document

f = Flow().add(uses='jinahub+docker://TransformerTorchEncoder')

doc = Document(content='my sentence to be encoded')

with f:
    f.post(on='/index', inputs=doc, on_done=lambda resp: print(resp.docs[0].embedding))
```


### Set `volumes`

With the `volumes` attribute, you can map the cache directory to your local cache directory, in order to avoid downloading 
the model each time you start the Flow.

```python
from jina import Flow

flow = Flow().add(
    uses='jinahub+docker://TransformerTorchEncoder',
    volumes='/your_home_folder/.cache/huggingface:/root/.cache/huggingface'
)
```

Alternatively, you can reference the docker image in the `yml` config and specify the `volumes` configuration.

`flow.yml`:

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://TransformerTorchEncoder'
    volumes: '/your_home_folder/.cache/huggingface:/root/.cache/huggingface'
```

And then use it like so:
```python
from jina import Flow

flow = Flow.load_config('flow.yml')
```


### Use other pre-trained models
You can specify the model to use with the parameter `pretrained_model_name_or_path`:
```python
from jina import Flow, Document

f = Flow().add(
    uses='jinahub+docker://TransformerTorchEncoder',
    uses_with={'pretrained_model_name_or_path': 'bert-base-uncased'}
)

doc = Document(content='this is a sentence to be encoded')

with f:
    f.post(on='/foo', inputs=doc, on_done=lambda resp: print(resp.docs[0].embedding))
```

You can check the supported pre-trained models [here](https://huggingface.co/transformers/pretrained_models.html)

### Use GPUs
To enable GPU, you can set the `device` parameter to a cuda device.
Make sure your machine is cuda-compatible.
If you're using a docker container, make sure to add the `gpu` tag and enable 
GPU access to Docker with `gpus='all'`.
Furthermore, make sure you satisfy the prerequisites mentioned in 
[Executor on GPU tutorial](https://docs.jina.ai/tutorials/gpu_executor/#prerequisites).

```python

from jina import Flow, Document

f = Flow().add(
    uses='jinahub+docker://TransformerTorchEncoder/gpu',
    uses_with={'device': 'cuda'}, gpus='all'
)

doc = Document(content='this is a sentence to be encoded')

with f:
    f.post(on='/foo', inputs=doc, on_done=lambda resp: print(resp.docs[0].embedding))
```

## Reference
- [huggingface](https://huggingface.co/models)
