# ✨ AudioCLIPImageEncoder

**AudioCLIPImageEncoder** is an encoder that encodes images using the [AudioCLIP](https://arxiv.org/abs/2106.13043) model.

This encoder is meant to be used in conjunction with the AudioCLIP text ([AudioCLIPTextEncoder](https://github.com/jina-ai/executors/tree/main/jinahub/encoders/text/AudioCLIPTextEncoder)) and audio ([AudioCLIPEncoder](https://github.com/jina-ai/executors/tree/main/jinahub/encoders/audio/AudioCLIPEncoder)) encoders, as it can embedd text, images and audio to the same latent space.

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


> These are only needed if you download the source code and directly use the class. Not needed if you use the Jina Hub method below.

First, you should download the model and the vocabulary, which will be saved into the `.cache` folder inside your current directory (will be created if it does not exist yet).

To do this, copy the `scripts/download_full.sh` script to your current directory and execute it:

```
wget https://raw.githubusercontent.com/jina-ai/executors/main/jinahub/encoders/image/AudioCLIPImageEncoder/scripts/download_full.sh && chmod +x download_full.sh
./download_full.sh
```

This will download the `Full` version of the model (this is the default model used by the executor). If you instead want to download the `Partial` version of the model, execute

```
wget https://raw.githubusercontent.com/jina-ai/executors/main/jinahub/encoders/image/AudioCLIPImageEncoder/scripts/download_partial.sh && chmod +x download_partial.sh
./download_partial.sh
```

And then you will also need to pass the argument `model_path='.cache/AudioCLIP-Partial-Training.pt'` when you initialize the executor.

## 🚀 Usages

### 🚚 Via JinaHub

#### Using docker image

Use the prebuilt images from JinaHub in your Python code: 

```python
from jina import Flow
	
f = Flow().add(
	uses='jinahub+docker://AudioCLIPImageEncoder',
	volumes='/path/to/pwd/.cache:/workspace/.cache'
)
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://AudioCLIPImageEncoder'
    volumes: '/path/to/pwd/.cache:/workspace/.cache'
```

#### Using source code

Use the source code from JinaHub in your Python code,

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://AudioCLIPImageEncoder')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://AudioCLIPImageEncoder'
```



## 🎉️ Example 

Here's a basic example demonstrating the use of this encoder

```python
import numpy as np
from jina import Flow, Document

f = Flow().add(
	uses='jinahub+docker://AudioCLIPImageEncoder',
	volumes='/path/to/pwd/.cache:/workspace/.cache'
)

with f:
	fake_image = np.ones((100, 100, 3), dtype=np.uint8)
	doc = Document(blob=fake_image)
	resp = f.post(on='foo', inputs=doc, return_results=True)
	print(resp[0])
```

#### Inputs 

`Document` with the `blob` attribute, where `blob` is an `np.ndarray` of dtype ``np.uint8`` (unless you set ``use_default_preprocessing=True``, then they can also be of a float type).

If you set `use_default_preprocessing=True` when creating this encoder, then the image arrays should have the shape `[H, W, C]`, and be in the RGB color format.

If you set `use_default_preprocessing=False` when creating this encoder, then you need to ensure that the images you pass in are already pre-processed. This means that they are all the same size (for batching) - the CLIP model was trained on `224 x 224` images, and that they are of the shape `[C, H, W]` (in the RGB color format). They should also be normalized.

#### Returns

`Document` with `embedding` field filled with an `ndarray` of the shape `(1024,)` with `dtype=nfloat32`.


## 🔍️ Reference

- [AudioCLIP paper](https://arxiv.org/abs/2106.13043)
- [AudioCLIP GitHub Repository](https://github.com/AndreyGuzhov/AudioCLIP)

