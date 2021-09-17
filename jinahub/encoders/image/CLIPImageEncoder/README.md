# CLIPImageEncoder

**CLIPImageEncoder** is an image encoder that wraps the image embedding functionality using the [CLIP](https://huggingface.co/transformers/model_doc/clip.html) model from huggingface transformers.
This encoder is meant to be used in conjunction with the [CLIPTextEncoder](https://hub.jina.ai/executor/livtkbkg),
as it can embed text and images to the same latent space.

Use the prebuilt images from JinaHub in your Python code,

```python
from jina import Flow

f = Flow().add(uses='jinahub+docker://CLIPImageEncoder',)
```

For more information on `gpu` usage and model `cache`, please checkout the [documentation](https://docs.jina.ai/tutorials/gpu-executor/).
For more information on CLIP model, please checkout the [blog post](https://openai.com/blog/clip/),
[paper](https://arxiv.org/abs/2103.00020) and [hugging face documentation](https://huggingface.co/transformers/model_doc/clip.html)