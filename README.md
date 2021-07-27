> ⚠️ Please do **not** commit your new Executor to this repository. This repository is **only** for Jina engineers to better manage in-house executors in a centralized way. You *may* submit PRs to fix bugs/add features to the existing ones.

> 🧭 To develop your own Executor, please use the [Executor Cookiecutter](https://github.com/jina-ai/cookiecutter-jina-executor/) and create your own Executor repo.

# Jina Executors

This repository provides a selection of [Executors](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Executor.md) for [Jina](https://github.com/jina-ai/jina).

**⚙️ Executor is how Jina processes Documents.** It is the building block of your Jina data pipeline, providing a specific functional needs: preparing data, encoding it with your model, storing, searching, and more.


## Usage

The following is general guidelines. Check each executor's README for details.

#### via Docker image

Use the prebuilt image from JinaHub in your Python code 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://ExecutorName')
```

#### via source code

Use the source code from JinaHub in your Python code:

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://ExecutorName')
```

<details>
<summary>Click here to see advance usage</summary>
	
### Via Pypi

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


### Via Docker

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

</details>	
	
## Contributing

**For internal Jina enigneers only:**

- add the new executor to the right subfolder.
	- crafters
	- [indexers](./jinahub/indexers) store and retrieve data
	- [encoders](./jinahub/encoders) compute the vector representation of data
	- rankers
- push your initial version to Jina Hub. Use the guide [here](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Hubble.md#2-push-and-pull-cli)
- add the UUID and secret to the secrets store. Make sure `(folder name) == (manifest alias) == (name in secrets store)` 

### Model downloading

Some Executors might require a large model. During CI/tests, it is advisable to download it as part of a fixture and store it to disk, to be re-used by the Executor.

In production, it is recommended to set up your workspace, model, and class to load from disk. If the Executor is served with Docker, make sure to also map the directory, as the Docker runtime does not get persisted.
