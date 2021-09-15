# CLIPImageEncoder

**CLIPImageEncoder** is an image encoder that wraps the image embedding functionality using the **CLIP** model from huggingface transformers.

It takes `Document`s with images stored in the `blob` attribute as inputs, and stores the
resulting image embedding in the `embedding` attribute. You can store original images in
the `blob` attribute, and they should be in the RGB format and have a shape `[H, W, 3]`). You
can also choose to pass in already pre-processed images (see the class documentation).

The **CLIP** model was originally proposed in [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020), and is trained to embed images and text to the same latent
space. The corresponding text encoder is **[CLIPTextEncoder](https://hub.jina.ai/executor/livtkbkg)**,
using both encoders together works well in multi-modal or cross-modal search applications.

## Usage

Here's a simple example of how to use CLIPImageEncoder in a Flow. We are embedding an image called `myimage.png`, stored in your working directory - this can be any image you like.

```python
import numpy as np
from jina import Flow, Document
from PIL import Image

f = Flow().add(uses='jinahub+docker://CLIPImageEncoder')

def print_result(resp):
    doc = resp.docs[0]
    print(f'Embedded image to {doc.embedding.shape[0]}-dimensional vector')

with f:
    doc = Document(blob=np.asarray(Image.open('myimage.png')))
    f.post(on='/foo', inputs=doc, on_done=print_result)
```

Note that this way the Executor will download the model every time it starts up. You can
re-use the cached model files by mounting the cache directory that the model is using
into the container. To do this, modify the Flow definition like this

```python
f = Flow().add(
    uses='jinahub+docker://CLIPImageEncoder',
    volumes='/your/home/dir/.cache/huggingface:/root/.cache/huggingface'
)
```

### With GPU

This encoder also offers a GPU version under the `gpu` tag. To use it, make sure to pass `device='cuda'`, as the initialization parameter, and `gpus='all'` when adding the containerized Executor to the Flow. See the [Executor on GPU](https://docs.jina.ai/tutorials/gpu_executor/) section of Jina documentation for more details.

Here's how you would modify the example above to use a GPU

```python
f = Flow().add(
    uses='jinahub+docker://CLIPImageEncoder',
    uses_with={'device': 'cuda'},
    gpus='all',
    volumes='/your/home/dir/.cache/huggingface:/root/.cache/huggingface' 
)
```

## Reference

- [CLIP blog post](https://openai.com/blog/clip/)
- [CLIP paper](https://arxiv.org/abs/2103.00020)
- [Huggingface transformers CLIP model documentation](https://huggingface.co/transformers/model_doc/clip.html)
