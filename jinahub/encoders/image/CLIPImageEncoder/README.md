# CLIPImageEncoder

**CLIPImageEncoder** is a class that wraps the image embedding functionality using the **CLIP** model from huggingface transformers.

The **CLIP** model originally was proposed in [Learning Transferable Visual Models From Natural Language Supervision](https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language_Supervision.pdf).

The following parameters can be passed on initialization:
- `pretrained_model_name_or_path`: Can be either:
    - A string, the model id of a pretrained CLIP model hosted
        inside a model repo on huggingface.co, e.g., 'openai/clip-vit-base-patch32'
    - A path to a directory containing model weights saved, e.g., ./my_model_directory/
- `base_feature_extractor`: Base feature extractor for images. 
      Defaults to ``pretrained_model_name_or_path`` if None
- `use_default_preprocessing`: Whether to use the `base_feature_extractor` on
        images (blobs) before encoding them. If you disable this, you must ensure
        that the images you pass in have the correct format, see the ``encode`` method
        for details.
- `device`: device that the model is on (should be "cpu", "cuda" or "cuda:X",
    where X is the index of the GPU on the machine)
- `default_batch_size`: fallback batch size in case there is no batch size sent in the request
- `default_traversal_paths`: fallback traversal path in case there is no traversal path sent in the request







## Usage 


```python
from jina import Flow, Document
import numpy as np

f = Flow().add(uses='jinahub+docker://CLIPImageEncoder')


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

[Documents](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Document.md) with `blob` of the shape `Height x Width x 3`. By default, the input `blob` must be an `ndarray` with `dtype=uint8`. The `Height` and `Width` can have arbitrary values.

If you set `use_default_preprocessing=True` when creating this encoder, then the image arrays should have the shape `[H, W, C]`, and be in the RGB color format.

If you set `use_default_preprocessing=False` when creating this encoder, then you need to ensure that the images you pass in are already pre-processed. This means that they are all the same size (for batching) - the CLIP model was trained on `224 x 224` images, and that they are of the shape `[C, H, W]` (in the RGB color format). They should also be normalized.

### Returns

[Documents](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Document.md) with `embedding` fields filled with an `ndarray` of the shape `512` with `dtype=nfloat32`.



## Reference

- [CLIP blog post](https://openai.com/blog/clip/)
- [CLIP paper](https://arxiv.org/abs/2103.00020)
- [Huggingface transformers CLIP model documentation](https://huggingface.co/transformers/model_doc/clip.html)

