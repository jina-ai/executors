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
- [indexers](./jinahub/indexers) store and retrieve data
- [encoders](./jinahub/encoders) compute the vector representation of data
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

## Contributing

If you want to develop your own Executor, please use the [Executor Cookiecutter](https://github.com/jina-ai/cookiecutter-jina-executor/) to start with. 

If you are an **external** user, this can then go into your own repository. Please do **not** commit to this repository. This is **only** for internal Jina Engineers. 

If you are a **Jina Engineer**, make sure to:
- add the new executor to the right subfolder. Check [Types](#types)
- push your initial version to Jina Hub. Use the guide [here](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Hubble.md#2-push-and-pull-cli)
- add the UUID and secret to the secrets store. Make sure `(folder name) == (manifest alias) == (name in secrets store)` 

### Model downloading

Some Executors might require a large model. During CI/tests, it is advisable to download it as part of a fixture and store it to disk, to be re-used by the Executor.

In production, it is recommended to set up your workspace, model, and class to load from disk. If the Executor is served with Docker, make sure to also map the directory, as the Docker runtime does not get persisted.

## Reference

- [Jina documentation](https://github.com/jina-ai/jina/tree/master/.github/2.0/cookbooks)
- [Jina examples](https://github.com/jina-ai/examples)