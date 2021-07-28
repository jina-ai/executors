# ✨ MinRanker

**MinRanker** is a class aggregates the score of the matched doc from the matched chunks.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [🌱 Prerequisites](#-prerequisites)
- [🚀 Usages](#-usages)
- [🎉️ Example](#%EF%B8%8F-example)
- [🔍️ Reference](#%EF%B8%8F-reference)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## 🌱 Prerequisites

Some conditions to fulfill before running the executor

## 🚀 Usages

### 🚚 Via JinaHub

#### using docker images
Use the prebuilt images from JinaHub in your python codes, 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://MinRanker')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://MinRanker'
```

#### using source codes
Use the source codes from JinaHub in your python codes,

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://MinRanker')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://MinRanker'
```


### 🐳 Via Docker

1. Clone the repo and build the docker image

	```shell
	git clone https://github.com/jina-ai/executors.git
	cd jinahub/rankers/
	docker build -t min-ranker .
	```

1. Use `min-ranker` in your codes

	```python
	from jina import Flow
	
	f = Flow().add(uses='docker://min-ranker:latest')
	```
	

## 🎉️ Example 

Here we **MUST** show a **MINIMAL WORKING EXAMPLE**. We recommend to use `jinahub+docker://MyDummyExecutor` for the purpose of boosting the usage of Jina Hub. 

It not necessary to demonstrate the usages of every inputs. It will be demonstrate in the next section.

```python
from jina import Flow, DocumentArray

f = Flow().add(uses='jinahub+docker://MinRanker')

with f:
    resp = f.post(on='search', inputs=DocumentArray(), return_results=True)
    print(f'{resp}')
```


## 🔍️ Reference

