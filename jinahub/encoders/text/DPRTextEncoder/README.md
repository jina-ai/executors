# DPRTextEncoder

 **DPRTextEncoder** is a class that encodes text into embeddings using the DPR model from huggingface transformers.

The **DPR** model was originally proposed in [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906).

This encoder supports both the DPR context and question encoders - you should specify which type you are using with the `encoder_type` parameter.

The following parameters can be passed on initialization:

- `pretrained_model_name_or_path`: Can be either:
	- the model id of a pretrained model hosted inside a model repo
		on huggingface.co.
	- A path to a directory containing model weights, saved using
		the transformers model's `save_pretrained()` method
- `encoder_type`: Either `'context'` or `'question'`. Make sure this
	matches the model that you are using.
- `base_tokenizer_model`: Base tokenizer model. The possible values are
	the same as for the ``pretrained_model_name_or_path`` parameters. If not
	provided, the ``pretrained_model_name_or_path`` parameter value will be used
- `title_tag_key`: The key under which the titles are saved in the documents'
    tag property. It is recommended to set this property for context encoders,
    to match the model pre-training. It has no effect for question encoders.
- `max_length`: Max length argument for the tokenizer
- `default_batch_size`: Default batch size for encoding, used if the
	batch size is not passed as a parameter with the request.
- `default_traversal_paths`: Default traversal paths for encoding, used if the
	traversal path is not passed as a parameter with the request.
- `device`: The device (cpu or gpu) that the model should be on.





## Usages

### Via JinaHub

Use the prebuilt images from JinaHub in your python codes, 

```python
from jina import Flow
	
f = Flow().add(
    uses='jinahub+docker://DPRTextEncoder',
)
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://DPRTextEncoder'
```

## Example 


```python
from jina import Flow, Document
import numpy as np
	
f = Flow().add(uses='jinahub+docker://DPRTextEncoder')

with f:
    f.post(
        on='/foo', 
        inputs=Document(text='your text'), 
        on_done=print
    )
```


### Inputs 

[Documents](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Document.md) with the [`text`](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Document.md#document-attributes) attribute. If you are using a context encoder the documents can additionally have a title tag, see initialization parameters.

### Returns

[Documents](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Document.md) with the `embedding` attribute filled with an `ndarray` of the shape `768` with `dtype=float32`.



## Reference

- [DPR paper](https://arxiv.org/abs/2004.04906)
- [Huggingface transformers DPR model documentation](https://huggingface.co/transformers/model_doc/dpr.html)
