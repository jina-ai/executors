# AudioCLIPEncoder

**AudioCLIPEncoder** is a class that wraps the [AudioCLIP](https://github.com/AndreyGuzhov/AudioCLIP) model for generating embeddings for audio data.

Before using it, please check the [prerequisites](#prerequisites).

This encoder is meant to be used in conjunction with the [AudioCLIPTextEncoder](https://hub.jina.ai/executor/jfe8kovq) and [AudioCLIPImageEncoder](https://hub.jina.ai/executor/3atsazub), as they embed text, images and audio to the same latent space.

You can either use the `Full` (where all three heads were trained) or the `Partial` (where the text and image heads were frozen) versions of the model.

For more information, such as how to run an Executor on a GPU, check [this guide](https://docs.jina.ai/tutorials/gpu-executor/).

## Prerequisites

> These are only needed if you use it with the `jinahub://AudioCLIPEncoder` syntax. 

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

And then you will also need to pass the argument `model_path='.cache/AudioCLIP-Partial-Training.pt'` when you initialize the executor, like so:

```python
with Flow().add(
    uses='jinahub://AudioCLIPEncoder',
    uses_with={
        'model_path': '.cache/AudioCLIP-Full-Training.pt'
    }
)
```

Replace 'Full' with 'Partial' if you downloaded that model.

## See also

- [AudioCLIPTextEncoder](https://hub.jina.ai/executor/jfe8kovq)
- [AudioCLIPImageEncoder](https://hub.jina.ai/executor/3atsazub)

## References

- [AudioCLIP paper](https://arxiv.org/abs/2106.13043)
- [AudioCLIP GitHub Repository](https://github.com/AndreyGuzhov/AudioCLIP)

<!-- version=v0.5 -->
