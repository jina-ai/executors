# AudioCLIPTextEncoder

**AudioCLIPTextEncoder** is an encoder that encodes text using the [AudioCLIP](https://arxiv.org/abs/2106.13043) model.

Before using it, please check the [prerequisites](#prerequisites).

This encoder is meant to be used in conjunction with the [AudioCLIPImageEncoder](https://hub.jina.ai/executor/3atsazub)
and [AudioCLIPEncoder](https://hub.jina.ai/executor/f4d22e1r) encoders, as they embed text, images and audio to the same
latent space.

You can either use the `Full` (where all three heads were trained) or the `Partial` (where the text and image heads were
frozen) versions of the model.

For more information, such as how to run an Executor on a GPU,
check [this guide](https://docs.jina.ai/tutorials/gpu-executor/).

## Prerequisites

> This should be met if 1) you are using `'jinahub+docker://'` syntax, or 2) leave `'download_model'` set to `False` (default value)

First, you should download the model and the vocabulary, which will be saved into the `.cache` folder inside your
current directory (will be created if it does not exist yet).

To do this, copy the `scripts/download_full.sh` script to your current directory and execute it:

```shell
wget https://raw.githubusercontent.com/jina-ai/executors/main/jinahub/encoders/image/AudioCLIPImageEncoder/scripts/download_full.sh && chmod +x download_full.sh
./download_full.sh
```

This will download the `Full` version of the model (this is the default model used by the executor. 
If you instead want to download the `Partial` version of the model, execute:

```shell
wget https://raw.githubusercontent.com/jina-ai/executors/main/jinahub/encoders/image/AudioCLIPImageEncoder/scripts/download_partial.sh && chmod +x download_partial.sh
./download_partial.sh
```

And then you will also need to pass the argument `model_path='.cache/AudioCLIP-Full-Training.pt'` when you initialize the executor, like so:

```python
with Flow().add(
        uses='jinahub://AudioCLIPTextEncoder',
        uses_with={
            'model_path': '.cache/AudioCLIP-Full-Training.pt'
        }
)
```

Replace 'Full' with 'Partial' if you downloaded that model.

### Usage within Docker

If you are using the Executor within Docker, you need to mount the local model directory and tell the Executor where to find it, like so:

```python
with Flow().add(
        uses='jinahub+docker://AudioCLIPTextEncoder',
        uses_with={
            'model_path': '/tmp/.cache/AudioCLIP-Full-Training.pt',
            'tokenizer_path': '/tmp/.cache/bpe_simple_vocab_16e6.txt.gz',
        },
        volumes='.cache:/tmp/.cache',
)
```

## See also

- [AudioCLIPImageEncoder](https://hub.jina.ai/executor/3atsazub)
- [AudioCLIPEncoder](https://hub.jina.ai/executor/f4d22e1r)``

## References

- [AudioCLIP paper](https://arxiv.org/abs/2106.13043)
- [AudioCLIP GitHub Repository](https://github.com/AndreyGuzhov/AudioCLIP)

<!-- version=v0.4 -->
