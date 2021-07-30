# ✨ CLIPTextEncoder

 **CLIPTextEncoder** is a class that wraps the text embedding functionality using the **CLIP** model from huggingface transformers

The **CLIP** model was originally proposed in  [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020).


The following parameters can be passed on initialization:
- `pretrained_model_name_or_path`: Can be either:
	- A string, the model id of a pretrained CLIP model hosted
              inside a model repo on huggingface.co, e.g., 'openai/clip-vit-base-patch32'
	- A path to a directory containing model weights, saved using
		the transformers model's `save_pretrained()` method
- `base_tokenizer_model`: Base tokenizer model.
        Defaults to ``pretrained_model_name_or_path`` if None
- `max_length`: Max length argument for the tokenizer.
        All CLIP models use 77 as the max length
- `device`: Device to be used. Use 'cuda' for GPU.
- `default_traversal_paths`: Default traversal paths for encoding, used if the
        traversal path is not passed as a parameter with the request.
- `default_batch_size`: Default batch size for encoding, used if the
        batch size is not passed as a parameter with the request.


**Table of Contents**

- [🌱 Prerequisites](#-prerequisites)
- [🚀 Usages](#-usages)
- [🎉️ Example](#%EF%B8%8F-example)
- [🔍️ Reference](#%EF%B8%8F-reference)


## 🌱 Prerequisites

> These are only needed if you download the source code and directly use the class. Not needed if you use the Jina Hub method below.

No prerequisites are required to run this executor.

## 🚀 Usages

### 🚚 Via JinaHub

Use the prebuilt images from JinaHub in your python codes, 

```python
from jina import Flow
	
f = Flow().add(
        uses='jinahub+docker://CLIPTextEncoder',
	)
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://CLIPTextEncoder'
```

## 🎉️ Example 


```python
from jina import Flow, Document
import numpy as np
	
f = Flow().add(
        uses='jinahub+docker://CLIPTextEncoder',
	)
	
def check_emb(resp):
    for doc in resp.data.docs:
        if doc.emb:
            assert doc.emb.shape == (512,)
	
with f:
	f.post(
	    on='/foo', 
	    inputs=Document(text='your text'), 
	    on_done=check_emb
	)
	    
```


### Inputs 

[Documents](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Document.md) with the [`text`](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Document.md#document-attributes) attribute.

### Returns

[Documents](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Document.md) with the `embedding` attribute filled with an `ndarray` of the shape `512` with `dtype=float32`.



## 🔍️ Reference

- [CLIP blog post](https://openai.com/blog/clip/)
- [CLIP paper](https://arxiv.org/abs/2103.00020)
- [Huggingface transformers CLIP model documentation](https://huggingface.co/transformers/model_doc/clip.html)