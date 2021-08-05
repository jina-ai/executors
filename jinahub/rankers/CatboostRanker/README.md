# ✨ CatboostRanker

**CatboostRanker** is a Jina ranker, using the [CatBoost](https://catboost.ai/) library, mode specifically, the `CatBoostRanker` for learning-to-rank.

`CatboostRanker` get `query_features`, `match_features` and `label` stored inside `Document` object from `DocumentArray`, and build a feature-label dataset to train the model.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [🌱 Prerequisites](#-prerequisites)
- [🚀 Usages](#-usages)
- [🎉️ Example](#%EF%B8%8F-example)
- [🔍️ Reference](#%EF%B8%8F-reference)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

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


### 📦️ Via Pypi

1. Install the `executors` package.

	```bash
	pip install git+https://github.com/jina-ai/executors/
	```

1. Use `executors` in your code

   ```python
   from jina import Flow
   from jinahub.rankers.CatboostRanker.catboost_ranker import CatboostRanker
   
   f = Flow().add(uses=CatboostRanker)
   ```


### 🐳 Via Docker

1. Clone the repo and build the docker image

	```shell
	git clone https://github.com/jina-ai/executors/
	cd jinahub/rankers/CatboostRanker
	docker build -t catboostranker .
	```

1. Use `catboostranker` in your codes

	```python
	from jina import Flow
	
	f = Flow().add(uses='docker://catboostranker:latest')
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
- `label` the tag name considered as groundtruth.
- `weight` (Optional)the tag name to store the importance of query, stored in each query document.
- `model_path` (Optional)the default model path to dump/load the document.
- `catboost_parameters` (Optional) the parameters to train catboost ranker, please refer to catboost documentation for more info.

### Inputs 

`Document`, with `tags` and matches with `tags`.

### Returns

Nothing


