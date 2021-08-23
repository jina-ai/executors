# ✨ AudioCLIPTextEncoder

**AudioCLIPTextEncoder** is an encoder that encodes text using the [AudioCLIP](https://arxiv.org/abs/2106.13043) model.

This encoder is meant to be used in conjunction with the [AudioCLIPImageEncoder](https://github.com/jina-ai/executors/tree/main/jinahub/encoders/image/AudioCLIPImageEncoder) and [AudioCLIPEncoder](https://github.com/jina-ai/executors/tree/main/jinahub/encoders/audio/AudioCLIPEncoder), so that text, images and audio are embedded to the same latent space.

You can use either the `Full` (where all three heads were trained) or the `Partial` (where the text and image heads were frozen) version of the model.

The following arguments can be passed on initialization:

- `model_path`: path to the pre-trained AudioCLIP model.
- `default_traversal_paths`: default traversal path (used if not specified in request's parameters)
- `default_batch_size`: default batch size (used if not specified in request's parameters)
- `device`: device that the model is on (should be "cpu", "cuda" or "cuda:X", where X is the index of the GPU on the machine)

#### Inputs 

`Document` with the `text` attribute.

#### Returns

`Document` with `embedding` fields filled with an `ndarray` of the shape 1024 with `dtype=nfloat32`.

**Table of Contents**

- [🌱 Prerequisites](#-prerequisites)
- [🚀 Usages](#-usages)
- [🎉️ Example](#-example)
- [🔍️ Reference](#-reference)

## 🌱 Prerequisites


First, you should download the model and the vocabulary, which will be saved into the `.cache` folder inside your current directory (will be created if it does not exist yet).

To do this, execute the following commands in your terminal

```
wget https://raw.githubusercontent.com/jina-ai/executors/main/jinahub/encoders/text/AudioCLIPTextEncoder/scripts/download_full.sh && chmod +x download_full.sh
./download_full.sh && rm download_full.sh
```

This will download the `Full` version of the model (this is the default model used by the executor). If you instead want to download the `Partial` version of the model, execute

```
wget https://raw.githubusercontent.com/jina-ai/executors/main/jinahub/encoders/text/AudioCLIPTextEncoder/scripts/download_partial.sh && chmod +x download_partial.sh
./download_partial.sh && rm download_partial.sh
```

And then you will also need to pass the argument `model_path='.cache/AudioCLIP-Partial-Training.pt'` when you initialize the executor.

## 🚀 Usages

### 🚚 Via JinaHub

#### Using docker images

Use the prebuilt images from JinaHub in your Python code, 

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

#### Using source code

Use the source code from JinaHub in your Python code,

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

## 🔍️ Reference

- [AudioCLIP paper](https://arxiv.org/abs/2106.13043)
- [AudioCLIP GitHub Repository](https://github.com/AndreyGuzhov/AudioCLIP)
