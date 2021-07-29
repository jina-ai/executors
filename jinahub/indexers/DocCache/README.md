# ✨ DocCache

DocCache is an Executor that can cache documents that it has seen before, by different combination of fields (or just one field). It then removes the Document that has the same combination of values in those fields from the DocumentArray, so it will not reach the following Executors in your Flow. 

This is useful for continuously indexing Documents, and not having to worry about indexing the same Document twice.

## Notes
The Executor only removes Documents in the `/index` endpoint. In the other endpoints, operations are done by the Document `id`.

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

## 🚀 Usages

### 🚚 Via JinaHub

#### using docker images
Use the prebuilt images from JinaHub in your Python code: 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://DocCache')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: indexer
    uses: 'jinahub+docker://DocCache'
```

#### using source code
Use the source code from JinaHub in your Python code:

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://DocCache')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: indexer
    uses: 'jinahub://DocCache'
```
<details>

### 📦️ Via Pypi

1. Install the `executor-indexers` package.

	```bash
	pip install git+https://github.com/jina-ai/executor-indexers/
	```

1. Use `executor-indexers` in your code

   ```python
   from jina import Flow
   from jinahub.indexers.DocCache import DocCache
   
   f = Flow().add(uses=DocCache)
   ```


### 🐳 Via Docker

1. Clone the repo and build the docker image

	```shell
	git clone https://github.com/jina-ai/executor-indexers/
	cd jinahub/indexers/cache
	docker build -t doc-cache-image .
	```

1. Use `doc-cache-image` in your code

	```python
	from jina import Flow
	
	f = Flow().add(uses='docker://doc-cache-image:latest')
	```
	
</details></details>

## 🎉️ Example 

In a Flow:

```python
from jina import Flow, DocumentArray, Document


docs = DocumentArray([
    Document(id=1, content='🐯'),
    Document(id=2, content='🐯'),
    Document(id=3, content='🐻'),
])

f = Flow(return_results=True).add(uses='jinahub+docker://DocCache')

with f:
    response = f.post(on='/index', inputs=docs, return_results=True)

    assert len(response[0].data.docs) == 2  # the duplicated Document is removed from the request
    assert set([doc.id for doc in response[0].data.docs]) == set(['1', '3'])

    docs_to_update = DocumentArray([
        Document(id=2, content='🐼')
    ])

    response = f.post(on='/update', inputs=docs_to_update, return_results=True)
    assert len(response[0].data.docs) == 1  # the Document with `id=2` is no longer duplicated.

    response = f.post(on='/index', inputs=docs[-1], return_results=True)
    assert len(response[0].data.docs) == 0  # the Document has been cached
    f.post(on='/delete', inputs=docs[-1])
    response = f.post(on='/index', inputs=docs[-1], return_results=True)
    assert len(response[0].data.docs) == 1  # the Document is cached again after the deletion
```

## Initialization
`fields` is the one or more [attributes of Document](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Document.md#document-attributes).
The value must be a tuple of strings (e.g. `[text, tags__author]`). The default value is `('content_hash', )`


## APIs

### `on='/index'`

This API calculates and caches the hash codes of the `Document`. If the Document has already previously cached,
it is removed from the `DocumentArray` and therefore no further Executor will receive it.

#### Inputs

`DocumentArray`. 

#### Outputs

`DocumentArray` without the duplicated `Document`.

### `on='/update'`

This API is used to update the hash codes of the cached `Document`. If the Document with the same `id` has already previously been cached, the hash code will be updated based on the new values of the `fields`

#### Inputs

`DocumentArray`.

#### Outputs

`DocumentArray` without the duplicated `Document`.

### `on='/delete'`

This API is used to delete the hash codes of the cached `Document`. If the Document with the same `id` has already previously been cached, the hash code will be deleted. 

#### Inputs

`DocumentArray`.

#### Outputs

`DocumentArray` without the duplicated `Document`.
