# ✨ LightGBMRanker

**LightGBMRanker** is a Jina ranker, using the [LightGBM](https://github.com/microsoft/LightGBM) library, mode specifically, the `LightGBMRanker` is used for learning-to-rank.

`LightGBMRanker` retrieves `query_features`, `match_features` and `relevance_label` stored inside `Document` object from `DocumentArray`, and builds a feature-label dataset to train the model.

**Table of Contents**

- [🌱 Prerequisites](#-prerequisites)
- [🚀 Usages](#-usages)
- [🎉️ Example](#%EF%B8%8F-example)
- [🔍️ Reference](#%EF%B8%8F-reference)


## 🌱 Prerequisites

- This Executor works on Python 3.7, 3.8 and 3.9. 
- While developing locally, make sure to install the [requirements](requirements.txt)
- Refer to LightGBM [documentation](https://github.com/microsoft/LightGBM/tree/master/examples/lambdarank) to learn how to use LightGBM to train a ranker.

## 🚀 Usages

### 🚚 Via JinaHub

#### using docker images

Use the prebuilt images from JinaHub in your Python code: 

```python
from jina import Flow
	
f = Flow().add(
    uses='jinahub+docker://LightGBMRanker',
    overwride_with={
        'query_features': ['query_price', 'query_size'],
        'match_features': ['match_price', 'match_size'],
        'relevance_level': 'relevance'
    }
)
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: ranker
    uses: 'jinahub+docker://LightGBMRanker'
    with:
      model_path: model.txt
      query_features: ['query_price', 'query_brand']
      match_features: ['match_price', 'match_brand']
      relevance_label: 'relevance'
```

#### using source code
Use the source code from JinaHub in your Python code:

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://LightGBMRanker',
               overwride_with={
                'query_features': ['query_price', 'query_size'],
                'match_features': ['match_price', 'match_size'],
                'relevance_level': 'relevance'
    }
)
```

The above code make use of the `tags` stored in `Document` and it's `matches`,
and create a feature value list for each query-match pair.
All features will be combined into a `np.ndarray` as training data.

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: ranker
    uses: 'jinahub://LightGBMRanker'
```


	

## 🎉️ Example 

```python
from jina import Flow, DocumentArray

f = Flow().add(uses='jinahub://LightGBMRanker')

da_to_train = DocumentArray()
da_to_search = DocumentArray()
# note, to see how we build a document array, please refer to tests/conftest.py

with f:
    f.post(on='/train', inputs=da_to_train)
    f.post(on='/search', inputs=da_to_search)
    f.post(on='/dump', parameters={'model_path': '/tmp/model.txt'})
    f.post(on='/load', parameters={'model_path': '/tmp/model.txt'})
```

Parameters:

- `query_features` the tag names to extract value from query document. Each query-match pair will be combined into a feature vector.
- `match_features` the tag names to extract value from match document. Each query-match pair will be combined into a feature vector.
- `relevance_label` the tag name to extract value from match to train the learning-to-rank model.
- `model_path` (Optional)the default model path to dump/load the document.
- `params` (Optional) the parameters to train catboost ranker, please refer to catboost documentation for more info.
- `categorical_query_features` (Optional) the tag names to extract value from query document as categorical features.
- `categorical_match_features` (Optional) the tag names to extract value from match document as categorical features.

### Inputs 

`Document`, with `tags` corresponded to `query_features` and matches with `tags` corresponded to `match_features` and `relevance_label`.

### Returns

Nothing
