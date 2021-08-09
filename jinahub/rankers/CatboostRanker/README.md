# ✨  CatboostRanker

**CatboostRanker** is a Jina ranker, using the [CatBoost](https://catboost.ai/) library, mode specifically, the `CatBoostRanker` for learning-to-rank.

`CatboostRanker` retrieves `query_features`, `match_features` and `relevance_label` stored inside `Document` object from `DocumentArray`, and builds a feature-label dataset to train the model.

**Table of Contents**

- [🌱 Prerequisites](#-prerequisites)
- [🚀 Usages](#-usages)
- [🎉️ Example](#%EF%B8%8F-example)
- [🔍️ Reference](#%EF%B8%8F-reference)


## 🌱 Prerequisites

- This Executor works on Python 3.7 and 3.8. 
- Make sure to install the [requirements](requirements.txt)
- Refer to CatBoost [tutorial](https://github.com/catboost/tutorials/blob/master/ranking/ranking_tutorial.ipynb) to learn how to use CatBoost to train a ranker.

## 🚀 Usages

### 🚚 Via JinaHub

#### using docker images

Use the prebuilt images from JinaHub in your Python code: 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://CatboostRanker')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: indexer
    uses: 'jinahub+docker://CatboostRanker'
```

#### using source code
Use the source code from JinaHub in your Python code:

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://CatboostRanker')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: ranker
    uses: 'jinahub://CatboostRanker'
```


	

## 🎉️ Example 

```python
from jina import Flow, DocumentArray

f = Flow().add(uses='jinahub://CatboostRanker')

da_to_train = DocumentArray()
da_to_search = DocumentArray()
# note, to see how we build a document array, please refer to tests/conftest.py

with f:
    f.post(on='/train', inputs=da_to_train)
    f.post(on='/search', inputs=da_to_search)
    f.post(on='/dump', parameters={'model_path': '/tmp/model.cbm'})
    f.post(on='/load', parameters={'model_path': '/tmp/model.cbm'})
```

Parameters:

- `query_features` the tag names to extract value from query document. Each query-match pair will be combined into a feature vector.
- `match_features` the tag names to extract value from match document. Each query-match pair will be combined into a feature vector.
- `relevance_label` the tag name to extract value from match to train the learning-to-rank model.
- `weight` (Optional)the tag name to store the importance of query, stored in each query document.
- `model_path` (Optional)the default model path to dump/load the document.
- `catboost_parameters` (Optional) the parameters to train catboost ranker, please refer to catboost documentation for more info.

### Inputs 

`Document`, with `tags` corresponded to `query_features` and matches with `tags` corresponded to `match_features` and `relevance_label`.

### Returns

Nothing
