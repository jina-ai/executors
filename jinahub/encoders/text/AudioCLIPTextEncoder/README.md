# ✨ AudioCLIPTextEncoder

**AudioCLIPTextEncoder** is an encoder that encodes text using the [AudioCLIP](https://arxiv.org/abs/2106.13043) model.

This encoder is meant to be used in conjunction with the AudioCLIP image and audio encoders, as it can embedd text, images and audio to the same latent space.

You can use either the `Full` (where all three heads were trained) or the `Partial` (where the text and image heads were frozen) version of the model.

The following arguments can be passed on initialization:

- `model_path`: path of the pre-trained AudioCLIP model.
- `default_traversal_paths`: default traversal path (used if not specified in request's parameters)
- `default_batch_size`: default batch size (used if not specified in request's parameters)
- `device`: device that the model is on (should be "cpu", "cuda" or "cuda:X", where X is the index of the GPU on the machine)

**Table of Contents**

- [🌱 Prerequisites](#-prerequisites)
- [🚀 Usages](#-usages)
- [🎉️ Example](#%EF%B8%8F-example)
- [🔍️ Reference](#%EF%B8%8F-reference)

## 🌱 Prerequisites

First, you should download the model and the vocabulary, which will be saved into the `.cache` folder inside your current directory (will be created if it does not exist yet).

To do this, copy the `scripts/download_full.sh` script to your current directory (and make it executable):

```
wget https://raw.githubusercontent.com/jina-ai/executors/main/jinahub/encoders/text/AudioCLIPTextEncoder/scripts/download_full.sh && chmod +x download_full.sh
./download_full.sh
```

This will download the `Full` version of the model (this is the default model used by the executor). If you instead want to download the `Partial` version of the model, execute

```
wget https://raw.githubusercontent.com/jina-ai/executors/main/jinahub/encoders/text/AudioCLIPTextEncoder/scripts/download_partial.sh && chmod +x download_partial.sh
./download_partial.sh
```

And then you will also need to pass the argument `model_path='.cache/AudioCLIP-Partial-Training.pt'` when you initialize the executor.

## 🚀 Usages

### 🚚 Via JinaHub

#### Using docker images

Use the prebuilt images from JinaHub in your python code, 

```python
from jina import Flow
	
f = Flow().add(
	uses='jinahub+docker://AudioCLIPTextEncoder',
	volumes='/path/to/pwd/.cache:/workspace/.cache'
)
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://AudioCLIPTextEncoder'
    volumes: '/path/to/pwd/.cache:/workspace/.cache'
```

#### Using source codes

Use the source code from JinaHub in your python code,

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://AudioCLIPTextEncoder')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://AudioCLIPTextEncoder'
```

<details>
<summary>Click here to see advance usage</summary>

### 📦️ Via pip

1. Install the `jinahub-audioclip-text` package.

	```bash
	pip install git+https://github.com/jina-ai/executor-text-audioclip-text-encoder.git
	```

1. Use `jinahub-audioclip-text` in your code

	```python
	from jina import Flow
	from jinahub.encoder.audioclip_text import AudioCLIPTextEncoder
	
	f = Flow().add(uses=AudioCLIPTextEncoder)
	```


### 🐳 Via Docker

1. Clone the repo and build the docker image

	```shell
	git clone https://github.com/jina-ai/executor-text-audioclip-text-encoder.git
	cd executor-text-audioclip-text-encoder
	docker build -t jinahub-audioclip-text .
	```

1. Use `jinahub-audioclip-text` in your codes

	```python
	from jina import Flow
	
	f = Flow().add(
		uses='docker://jinahub-audioclip-text:latest',
		volumes='/path/to/pwd/.cache:/workspace/.cache'
	)
	```
	
</details>

## 🎉️ Example 

Here's a basic example demonstrating the use of this encoder

```python
from jina import Flow, Document

f = Flow().add(
	uses='jinahub+docker://AudioCLIPTextEncoder',
	volumes='/path/to/pwd/.cache:/workspace/.cache'
)

with f:
	doc = Document(text='test text')
    resp = f.post(on='foo', inputs=doc, return_results=True)
    print(resp[0])
```

#### Inputs 

`Document` with the `text` attribute.

#### Returns

`Document` with `embedding` fields filled with an `ndarray` of the shape 1024 with `dtype=nfloat32`.


## 🔍️ Reference

- [AudioCLIP paper](https://arxiv.org/abs/2106.13043)
- [AudioCLIP GitHub Repository](https://github.com/AndreyGuzhov/AudioCLIP)

