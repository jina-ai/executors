# AudioCLIPImageEncoder

**AudioCLIPImageEncoder** is an encoder that encodes images using the [AudioCLIP](https://arxiv.org/abs/2106.13043) model.

This encoder is meant to be used in conjunction with the AudioCLIP text ([AudioCLIPTextEncoder](https://github.com/jina-ai/executors/tree/main/jinahub/encoders/text/AudioCLIPTextEncoder)) and audio ([AudioCLIPEncoder](https://github.com/jina-ai/executors/tree/main/jinahub/encoders/audio/AudioCLIPEncoder)) encoders, as it can embedd text, images and audio to the same latent space.

You can use either the `Full` (where all three heads were trained) or the `Partial` (where the text and image heads were frozen) version of the model.

The following arguments can be passed on initialization:

- `model_path`: path of the pre-trained AudioCLIP model.
- `default_traversal_paths`: default traversal path (used if not specified in request's parameters)
- `default_batch_size`: default batch size (used if not specified in request's parameters)
- `device`: device that the model is on (should be "cpu", "cuda" or "cuda:X", where X is the index of the GPU on the machine)


## Prerequisites

> These are only needed if you use `jinahub://AudioCLIPTextEncoder`. 

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


## Reference

- [AudioCLIP paper](https://arxiv.org/abs/2106.13043)
- [AudioCLIP GitHub Repository](https://github.com/AndreyGuzhov/AudioCLIP)
