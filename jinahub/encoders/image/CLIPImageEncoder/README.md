# CLIPImageEncoder

**CLIPImageEncoder** is an image encoder that wraps the image embedding functionality using the [CLIP](https://huggingface.co/transformers/model_doc/clip.html) model from huggingface transformers.

This encoder is meant to be used in conjunction with the [CLIPTextEncoder](https://hub.jina.ai/executor/livtkbkg),

as it can embed text and images to the same latent space.


## Usage

Use the prebuilt images from JinaHub in your Python code,

```python
from jina import Flow, Document

f = Flow().add(
    uses='jinahub+docker://CLIPImageEncoder',
)
```

or in the .yml config.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://CLIPImageEncoder'

```

Note that this way the Executor will download the model every time it starts up. You can
re-use the cached model files by mounting the cache directory that the model is using
into the container. To do this, modify the Flow definition like this

```python
from jina import Flow

f = Flow().add(
    uses='jinahub+docker://CLIPImageEncoder',
    volumes='/your/home/dir/.cache/huggingface:/root/.cache/huggingface'
)
```

This encoder also offers a GPU version under the `gpu` tag. To use it, make sure to pass `device='cuda'`, as the initialization parameter, and `gpus='all'` when adding the containerized Executor to the Flow. See the [Executor on GPU](https://docs.jina.ai/tutorials/gpu_executor/) section of Jina documentation for more details.

Here's how you would modify the example above to use a GPU

```python
from jina import Flow

f = Flow().add(
    uses='jinahub+docker://CLIPImageEncoder/gpu',
    uses_with={'device': 'cuda'},
    gpus='all',
    volumes='/your/home/dir/.cache/huggingface:/root/.cache/huggingface' 
)
```

## Reference

- [CLIP blog post](https://openai.com/blog/clip/)
- [CLIP paper](https://arxiv.org/abs/2103.00020)
- [Huggingface transformers CLIP model documentation](https://huggingface.co/transformers/model_doc/clip.html)