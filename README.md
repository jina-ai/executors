<p align="center">
<img src="https://github.com/jina-ai/jina/blob/master/.github/logo-only.gif?raw=true" alt="Jina logo: Jina is a cloud-native neural search framework" width="200px">
</p>

# ✨ Jina Executors

This repository provides a selection of [Executors](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Executor.md) for [Jina 2.0](https://github.com/jina-ai/jina).

⚙️ Executors are the building blocks of your Jina data pipeline, providing a specific functional needs: preparing data, encoding it with your model, storing, searching, and more.

To get started with Jina 2.0 check the guide [here](https://github.com/jina-ai/jina#run-quick-demo)

## Types 

They are structured into folders, by type. Check out the README documentation by type, or by individual Executor.

We provide the following types of Executors:

- crafters
- [indexers](./jinahub/indexers)
- encoders
- rankers

## 🚀 Usage

The following is general guidelines. Check each executor's README for details.

### 🚚 Via JinaHub

#### using Docker images

Use the prebuilt images from JinaHub in your Python code 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://ExecutorName')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: indexer
    uses: 'jinahub+docker://ExecutorName'
``` 

#### using source code

Use the source code from JinaHub in your Python code:

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://ExecutorName')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: indexer
    uses: 'jinahub://ExecutorName'
```


### 📦️ Via Pypi

1. Install the `executors` package.

	```bash
	pip install git+https://github.com/jina-ai/executors/
	```

1. Use `executors` in your code

   ```python
   from jina import Flow
   from jinahub.type.subtype.ExecutorName import ExecutorName
   
   f = Flow().add(uses=ExecutorName)
   ```


### 🐳 Via Docker

1. Clone the repo and build the docker image

	```shell
	git clone https://github.com/jina-ai/executors
	cd executors/type/subtype
	docker build -t executor-image .
	```

1. Use `executor-image` in your code

	```python
	from jina import Flow
	
	f = Flow().add(uses='docker://executor-image:latest')
	```
	

## Reference

- [Jina documentation](https://github.com/jina-ai/jina/tree/master/.github/2.0/cookbooks)
- [Jina examples](https://github.com/jina-ai/examples)