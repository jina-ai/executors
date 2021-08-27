# Contributing to Jina executors

While this repository is primarily developed and maintained by the Jina engineering team, we welcome contributions from the community as well!

  - [📄 General guidelines](#-general-guidelines)
  - [⚙️ Adding a new executor](#️-adding-a-new-executor)
    - [🧰 Requirements and Dockerfile](#-requirements-and-dockerfile)
      - [GPU Dockerfile (🚧 WIP)](#gpu-dockerfile--wip)
    - [✨ Coding standards](#-coding-standards)
    - [📖 Documentation](#-documentation)
    - [💾 Downloading large artifacts](#-downloading-large-artifacts)
    - [🛠️ Writing tests](#️-writing-tests)
      - [Test-time dependencies](#test-time-dependencies)
      - [Integration tests](#integration-tests)
      - [Unit tests](#unit-tests)
      - [GPU tests (🚧 WIP)](#gpu-tests--wip)
    - [📦 Uploading to JinaHub](#-uploading-to-jinahub)

## 📄 General guidelines

We welcome all contributions, be they bug reports, feature (new executor) requests, or PRs.

If your intend to submit a PR which introduces new features or new executors, please open an issue beforehand, so that we can agree on the best course of action. For PRs that introduce minor bugfixes or documentation improvements this is not necessary.

For general instructions on how to open a PR, and how to write good commit messages, please see Jina's [contribution guide](https://github.com/jina-ai/jina/blob/master/CONTRIBUTING.md).

## ⚙️ Adding a new executor

Before adding a new executor, please read the [Executor cookbook](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Executor.md) to understand how the Executor should work.

Naming convention for executors is roughly `NameSubtypeType` - for example, if you want to create a new *text* (subtype) *encoder* (type) based on the *BERT* model (name), you would name it `BERTTextEncoder`. Sometimes there is no subtype, for example `FaissSearcher`. The name has to be in camel case, and will be used both for the name of the executor's directory, and for the name of the executor's class.

To create a new executor, create a new folder in the appopriate location in this repository (for example, if you want to create a new text encoder, you would create a folder iniside `jinahub/encoders/text`). You can do this using the `jina hub new` command, as this will also create all the necessary auxiliary files (readmes, `config.yml`, `manifest.yml`, etc.)


The module's python code should be structured according to [Structure of the repository](#https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Executor.md#structure-of-the-repository) chapter in the cookbook - a single python file if it's something simple, or a python package for all other cases.

### 🧰 Requirements and Dockerfile

For all the python dependencies of your executor, use a `requirements.txt` file at the base of your repository. Do not put `jina` there - when the executor is used in a docker container or through Jina hub, `jina` will already be installed. Although you will need `jina` installed in your local development environment, of course - see [test requirements](#test-time-dependencies) for that.

Usually, this is all you need. In this case you do not need to create a Dockerfile - when uploading to the hub, one will be created automatically.

If there are any other requirements (such as installing system packages using `apt-get`), then you need to create a `Dockerfile` with instructions for installation of all other requirements. Some things to take into account when creating the Dockerfile:
- If possible, use `jinaai/jina:2-py37-perf` as your base image. It is a lightweight image, based on the `python:3.7-slim` image, and comes with the latest stable version of `jina` preinstalled.
- The entrypoint should be
    ```
    ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
    ```

#### GPU Dockerfile (🚧 WIP)


### ✨ Coding standards

To ensure the readability of our code, we stick to a few conventions:
- We format python files using `black`, with the exception that we using single `'quotes'`, instead of double `"quotes"`.
- For linting, we use `flake8`.
- For sorting imports, we use `isort`.

The `setup.cfg` and `pyproject.toml` already contain the proper configuration for these tools. If you are working in a moren IDE (such a VSCode), which integrates these tools, the options will be picked up.

If you are working from the command line, you should use these tools form the root of this repository (and not from the executor's folder), for example
```
black jinahub/encoders/text/BERTTextEncoder
```

### 📖 Documentation

When writing code you should already be adding documentation, by writing docstrings for all classes, methods and functions. Type hints for all arguments and return values should be added as well.

On top of that, you should also fill in the `README.md`. A standard structure of this file consists of (in this order):

- A short description of what your executor is and what it does
- Description of the initialization arguments (can be copied from the docstring)
- Description of inputs for the request method(s).
- Description of outputs from the request method(s).
- Pre-requisites, of which there are two types:
    - How to set up the environment and install requirements when developing the executor locally
    - How to download large artifacts (more on that below). These instructions should contain copy-pastable command line instructions.
- Instructions on how to use the executor using `jinahub+docker://` or `jinahub://`, using either a python script to set up the flow manually, or a `flow.yaml` file.
- A short pure-python example, using `jinahub+docker://` - should be self contained and runnable. Also meaningful, if possible.
-  A References section, with links to relevant references (papers, blog posts, etc.)

### 💾 Downloading large artifacts

Often you will find that for the executor to work, a large file needs to be downloaded first - usually this would be a file with pre-trained model weights. If this is done at the start of the executor, it will lead to really long startup times, or even timeouts, which will frustrate users. 

If this file is baked in the docker image (not even possible in all cases), this will create overly large docker images, and prevent users for optimizing storage in cases where they want to run multiple instances of the executor on the same machine, among other things.

The solution in this case is to instruct users how to download the file **before** starting the executor, and then use that file in their executor. So for this case, you should:
- Add simple copy-pastable instructions on how to download the large files to the readme
- Add instructions on how to specify the path to the file at executor initialization (if needed) and how to mount the file to a Docker container in the readme
- If the file path is not provided, or the file doesn't exist, add an error telling the user that file needs to be downloaded, and pointing them to the readme for further instructions.

### 🛠️ Writing tests

To make sure your executor is working as it should, we need tests. We aim for 100% test coverage - that is, for all the code to be covered by tests. Tests should live inside the `tests/` folder, and should be written for the [pytest](https://docs.pytest.org/en/6.2.x/) test suite. There are two kinds of tests that you need: **integration** and **unit**. So the structure of the tests directory should look like this

```
tests/
├── unit/             # Required
├── integration/      # Required
├── conftest.py       # Required, see integration tests
├── requirements.txt  # Required, see test-time dependencies
└── pre_test.sh       # Optional, see test-time dependencies

```

#### Test-time dependencies

To make sure that all the dependencies needed by the executor and any test time dependencies are properly installed, you need to create these files:

- `tests/requirements.txt` (required): Put any python requirements needed for testing, but not covered in `requirements.txt`, in this file - you will need `pytest`, for example. Do not list `jina` here - in CI, the latest `2.x` version will be automatically installed (and you should install it locally as well).
- `tests/pre_test.sh` (optional): If you need any system dependencies installed before performing the tests, put it in this script. This will be run before any dependencies are installed.

If your executors requires downloading large files (see [below](#-downloading-large-artifacts)), do this with a pytest fixture. Here's an example

```python
@pytest.fixture(scope="session", autouse=True)
def download_files():
    download()
    yield
    cleanup()  # Not strictly required
```

#### Integration tests

In integration tests you need to test the use of the executor in a jina Flow. These tests should be simple (and not many, one will suffice in most situations): you send some documents through the flow which contains the executor, and check that the results are what you expect. Here's an example for a text encoder:

```python
@pytest.mark.parametrize("request_size", [1, 10, 50, 100])
def test_integration(data_generator: Callable, request_size: int):
    with Flow(return_results=True).add(uses=AudioCLIPTextEncoder) as flow:
        resp = flow.post(
            on="/index",
            inputs=data_generator(),
            request_size=request_size,
            return_results=True,
        )

    assert sum(len(resp_batch.docs) for resp_batch in resp) == 50
    for r in resp:
        for doc in r.docs:
            assert doc.embedding is not None
            assert doc.embedding.shape == (1024,)
```

Also, test that the executor can be run from a docker container, when running it with `jina executor --uses=docker://...`. This test will look like this

```python
import subprocess

import pytest

@pytest.mark.docker
def test_docker_runtime(build_docker_image: str):
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(
            ['jina', 'executor', '--uses=docker://{build_docker_image}'], 
            timeout=30,
            check=True
        )
```

And the needed fixture looks like this (put it in `tests/conftest.py`)

```python
@pytest.fixture(scope='session')
def build_docker_image() -> str:
    img_name = Path(__file__).parents[1].stem.lower() # lower case name of executor dir
    subprocess.run(['docker', 'build', '-t', img_name, '.'], check=True)

    return img_name
```

Here replace `myexecutor` with the *lower case* name of your executor (the image with its name is built in CI before the test is run). You can also add additional arguments to the main command - if you need to download large files for your model (which should have been done in a fixture at test time), you would add `'--volumes=/path/to/file:/path/to/file/in/container/'`.

What this test does is to launch the executor in a docker container, and if no other errors occur, timeout after 30 seconds (more than enough time for the executor to initialize), which means that it was launched succesfully. On error you will see the full printout of the output in the container, so that you can debug the issue.

The test is given the `@pytest.mark.docker` decorator, so that when testing locally, you can mostly skip this test - because otherwise the image will be re-built after every change, which takes time. You can do this by running pytest as

```
pytest -m "not docker"
```

#### Unit tests

Unit tests test the functioning of your executor. These tests need to be detailed - you want to test everything here, including possible edge cases and errors. Here's a list of things that you need to do:
- Test that the executor can be loaded from `config.yaml` using `Executor.load_config`
- Test that requests work with **all** allowed inputs: this includes `None`, an empty `DocumentArray`, and allowed inputs (a `DocumentArray` with `Document` elements). For the last one, check also what happens when (some) documents do not have the required attribute, e.g. `text` or `blob`.
- Test that warnings and errors are raised (or logged) when they should be.
- Test that the values that can be passed to `parameters` in requests have the desired effect 
- For encoders: check that the resulting embeddings have the desired **semantic similarity** properties. This means, for example, that for an image encoder embeddings of the images of a cat and dog should be closer to each other than to an image of an airplane. Here you can look at other encoders of the same modality and copy this test from them. 


### GPU tests (🚧 WIP)

### 📦 Uploading to JinaHub

> ⚠️ This step should only be performed by Jina engineers!

When the executor is complete, it is time to publish it to Jina hub (do this before merging the PR). This should only be done by Jina engineers! If you are a community contributor, ask the Jina employeed supervising your PR to do it.

Before pushing anything, make sure that the `name` field in `manifest.yml` matches the name of the executor's folder. To push the executor to JinaHub, follow [these instructions](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Hub.md#2-push-executor-to-jinahub) from the Hub cookbook. Once you do that, add the UUID and secret to the executors secrets storage.

And that's it. Once this is done, the executor will automatically be pushed to JinaHub each time its PR is merged.
