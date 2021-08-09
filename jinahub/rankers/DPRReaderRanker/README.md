# ✨  DPRReaderRanker

 **DPRReaderRanker** Performs a re-ranking of the matches for each document (question), as well as the answer spans extraction for each match. It uses the DPR Reader model to re-rank documents based on cross-attention between the question and the potential answer passages.

The **DPR** model was originally proposed in [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906).

The following parameters can be passed on initialization:

- `pretrained_model_name_or_path`: Can be either:
    - the model id of a pretrained model hosted inside a model repo
        on huggingface.co.
    - A path to a directory containing model weights, saved using
        the transformers model's `save_pretrained()` method
- `base_tokenizer_model`: Base tokenizer model. The possible values are the 
    same as for the `pretrained_model_name_or_path` parameters. If not provided,
    the `pretrained_model_name_or_path` parameter value will be used
- `title_tag_key`: The key of the tag that contains document title in the
        match documents. Specify it if you want the text of the matches to be combined
        with their titles (to mirror the method used in training of the original model)
- `num_spans_per_match`: Number of spans to extract per match
- `max_length`: Max length argument for the tokenizer
- `default_batch_size`: Default batch size for processing documents, used if the
    batch size is not passed as a parameter with the request.
- `default_traversal_paths`: Default traversal paths for processing documents,
    used if the traversal path is not passed as a parameter with the request.
- `device`: The device (cpu or gpu) that the model should be on.


**Table of Contents**

- [🌱 Prerequisites](#-prerequisites)
- [🚀 Usages](#-usages)
- [🎉️ Example](#%EF%B8%8F-example)
- [🔍️ Reference](#%EF%B8%8F-reference)

## 🌱 Prerequisites

No prerequisites are required to run this executor.

## 🚀 Usages

### 🚚 Via JinaHub

Use the prebuilt images from JinaHub in your python codes, 

```python
from jina import Flow
	
f = Flow().add(
    uses='jinahub+docker://DPRReaderRanker',
)
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://DPRReaderRanker'
```

## 🎉️ Example 


```python
from jina import Flow, Document
	
f = Flow().add(uses='jinahub+docker://DPRReaderRanker')

doc = Document(text='What is Jina?')
match = Document(
    text='Jina AI is a Neural Search Company, enabling cloud-native neural'
    ' search powered by state-of-the-art AI and deep learning',
    tags={'title': 'Jina AI'},
)
doc.matches.append(match)

with f:
    f.post(
        on='/foo', 
        inputs=doc, 
        on_done=print
    )
```


### Inputs 

[Documents](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Document.md) with the [`text`](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Document.md#document-attributes) attribute, and with matches that themselves also have a `text` attribute (and optionally a title tag, see initialization parameters).

### Returns

[Documents](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Document.md) with new matches, where the `text` attribute of those matches is taken from the best-scoring spans from `text` of the original matches.



## 🔍️ Reference

- [DPR paper](https://arxiv.org/abs/2004.04906)
- [Huggingface transformers DPR model documentation](https://huggingface.co/transformers/model_doc/dpr.html)
