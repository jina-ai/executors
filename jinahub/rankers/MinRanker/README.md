# 📝 PLEASE READ [THE GUIDELINES](.github/GUIDELINES.md) BEFORE STARTING.

# 🏗️ PLEASE CHECK OUT [STEP-BY-STEP](.github/STEP_BY_STEP.md)

----

# ✨ MyDummyExecutor

**MyDummyExecutor** is a class that ...

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
	
f = Flow().add(uses='jinahub+docker://MyDummyExecutor')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://MyDummyExecutor'
```

#### using source codes
Use the source codes from JinaHub in your python codes,

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://MyDummyExecutor')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://MyDummyExecutor'
```


### 📦️ Via Pypi

1. Install the `jinahub-MY-DUMMY-EXECUTOR` package.

	```bash
	pip install git+https://github.com/jina-ai/EXECUTOR_REPO_NAME.git
	```

1. Use `jinahub-MY-DUMMY-EXECUTOR` in your code

	```python
	from jina import Flow
	from jinahub.SUB_PACKAGE_NAME.MODULE_NAME import MyDummyExecutor
	
	f = Flow().add(uses=MyDummyExecutor)
	```


### 🐳 Via Docker

1. Clone the repo and build the docker image

	```shell
	git clone https://github.com/jina-ai/EXECUTOR_REPO_NAME.git
	cd EXECUTOR_REPO_NAME
	docker build -t my-dummy-executor-image .
	```

1. Use `my-dummy-executor-image` in your codes

	```python
	from jina import Flow
	
	f = Flow().add(uses='docker://my-dummy-executor-image:latest')
	```
	

## 🎉️ Example 

Here we **MUST** show a **MINIMAL WORKING EXAMPLE**. We recommend to use `jinahub+docker://MyDummyExecutor` for the purpose of boosting the usage of Jina Hub. 

It not necessary to demonstrate the usages of every inputs. It will be demonstrate in the next section.

```python
from jina import Flow, Document

f = Flow().add(uses='jinahub+docker://MyDummyExecutor')

with f:
    resp = f.post(on='foo', inputs=Document(), return_results=True)
    print(f'{resp}')
```

### `on=/index` (Optional)

When there are multiple APIs, we need to list the inputs and outputs for each one. If there is only one universal API, we only demonstrate the inputs and outputs for it.

#### Inputs 

`Document` with `blob` of the shape `256`.

#### Returns

`Document` with `embedding` fields filled with an `ndarray` of the shape `embedding_dim` (=128, by default) with `dtype=nfloat32`.

### `on=/update` (Optional)

When there are multiple APIs, we need to list the inputs and outputs for each on

#### Inputs 

`Document` with `blob` of the shape `256`.

#### Returns

`Document` with `embedding` fields filled with an `ndarray` of the shape `embedding_dim` (=128, by default) with `dtype=nfloat32`.

## 🔍️ Reference
- Some reference

