# CLIPTextEncoder

**CLIPTextEncoder** is a text encoder that wraps the text embedding functionality using the **CLIP** model from huggingface transformers.

It takes `Document`s with text stored in the `text` attribute as inputs, and stores the
resulting embedding in the `embedding` attribute.

The **CLIP** model was originally proposed in [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020), and is trained to embed images and text to the same latent
space. The corresponding image encoder is **[CLIPImageEncoder](https://hub.jina.ai/executor/0hnlmu3q)**,
using both encoders together works well in multi-modal or cross-modal search applications.

## Usage

Here's a simple example of how to use CLIPTextEncoder in a Flow

```python
from jina import Flow, Document

f = Flow().add(uses='jinahub+docker://CLIPTextEncoder')

def print_result(resp):
    doc = resp.docs[0]
    print(f'Embedded "{doc.text}" to {doc.embedding.shape[0]}-dimensional vector')

with f:
    doc = Document(text='your text')
    f.post(on='/foo', inputs=doc, on_done=print_result)
```

Note that this way the Executor will download the model every time it starts up. You can
re-use the cached model files by mounting the cache directory that the model is using
into the container. To do this, modify the Flow definition like this

```python
f = Flow().add(
    uses='jinahub+docker://CLIPTextEncoder',
    volumes='/your/home/dir/.cache/huggingface:/root/.cache/huggingface'
)
```

### With GPU

This encoder also offers a GPU version under the `gpu` tag. To use it, make sure to pass `device='cuda'`, as the initialization parameter, and `gpus='all'` when adding the containerized Executor to the Flow. See the [Executor on GPU](https://docs.jina.ai/tutorials/gpu_executor/) section of Jina documentation for more details.

Here's how you would modify the example above to use a GPU

```python
f = Flow().add(
    uses='jinahub+docker://CLIPTextEncoder/gpu',
    uses_with={'device': 'cuda'},
    gpus='all',
    volumes='/your/home/dir/.cache/huggingface:/root/.cache/huggingface' 
)
```

## Reference

- [CLIP blog post](https://openai.com/blog/clip/)
- [CLIP paper](https://arxiv.org/abs/2103.00020)
- [Huggingface transformers CLIP model documentation](https://huggingface.co/transformers/model_doc/clip.html)
- [AudioCLIPEncoder](https://hub.jina.ai/executor/f4d22e1r)
- [AudioCLIPTextEncoder](https://hub.jina.ai/executor/jfe8kovq)
- [AudioCLIPImageEncoder](https://hub.jina.ai/executor/3atsazub)