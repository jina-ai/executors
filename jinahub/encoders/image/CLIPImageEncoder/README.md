# ✨ CLIPImageEncoder

**CLIPImageEncoder** is a class that wraps the image embedding functionality using the **CLIP** model from huggingface transformers.

The **CLIP** model originally was proposed in [Learning Transferable Visual Models From Natural Language Supervision](https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language_Supervision.pdf).

The following parameters can be passed on initialization:
- `pretrained_model_name_or_path`: Can be either:
    - A string, the model id of a pretrained CLIP model hosted
        inside a model repo on huggingface.co, e.g., 'openai/clip-vit-base-patch32'
    - A path to a directory containing model weights saved, e.g., ./my_model_directory/
- `use_default_preprocessing`: Whether to use the default preprocessing on
        images (blobs) before encoding them. If you disable this, you must ensure
        that the images you pass in have the correct format, see the ``encode`` method
        for details.
- `device`: device that the model is on (should be "cpu", "cuda" or "cuda:X",
    where X is the index of the GPU on the machine)
- `default_batch_size`: fallback batch size in case there is no batch size sent in the request
- `default_traversal_paths`: fallback traversal path in case there is no traversal path sent in the request


**Table of Contents**

- [🌱 Prerequisites](#-prerequisites)
- [🚀 Usages](#-usages)
- [🎉️ Example](#-example)
- [🔍️ Reference](#-reference)


## 🌱 Prerequisites

> These are only needed if you download the source code and directly use the class. Not needed if you use the Jina Hub method below.

In case you want to install the dependencies locally run 
```
pip install -r requirements.txt
```

## 🚀 Usages

### 🚚 Via JinaHub

#### using docker images

Use the prebuilt images from JinaHub in your Python code: 

```python
from jina import Flow
	
f = Flow().add(
        uses='jinahub+docker://CLIPImageEncoder',
        volumes='/your_home_folder/.cache/clip:/root/.cache/clip')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://CLIPImageEncoder'
    volumes: '/your_home_folder/.cache/clip:/root/.cache/clip'
```

#### using source code
Use the source code from JinaHub in your Python code:

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://CLIPImageEncoder',
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://CLIPImageEncoder'
```


## 🎉️ Example 


```python
from jina import Flow, Document
import numpy as n

f = Flow().add(uses='jinahub+docker://CLIPImageEncoder', )


def check_resp(resp):
    for _doc in resp.data.docs:
        doc = Document(_doc)
        print(f'embedding shape: {doc.embedding.shape}')


with f:
    f.post(on='/foo',
           inputs=Document(blob=np.ones((800, 224, 3), dtype=np.uint8)),
           on_done=check_resp)
	    
```


### Inputs 

[Documents](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Document.md) with `blob` of the shape `Height x Width x 3`. By default, the input `blob` must be an `ndarray` with `dtype=uint8`. The `Height` and `Width` can have arbitrary values. When setting `use_default_preprocessing=False`, the input `blob` must have the size of `224x224x3` with `dtype=float32`.

### Returns

[Documents](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Document.md) with `embedding` fields filled with an `ndarray` of the shape `512` with `dtype=nfloat32`.



## 🔍️ Reference

- [CLIP blog post](https://openai.com/blog/clip/)
- [CLIP paper](https://arxiv.org/abs/2103.00020)
- [Huggingface transformers CLIP model documentation](https://huggingface.co/transformers/model_doc/clip.html)

