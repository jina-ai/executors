# AudioCLIPEncoder

**AudioCLIPEncoder** is a class that wraps the [AudioCLIP](https://github.com/AndreyGuzhov/AudioCLIP) model for generating embeddings for audio data.

This encoder is meant to be used in conjunction with the [AudioCLIPImageEncoder](https://github.com/jina-ai/executors/tree/main/jinahub/encoders/image/AudioCLIPImageEncoder) and [AudioCLIPTextEncoder](https://github.com/jina-ai/executors/tree/main/jinahub/encoders/text/AudioCLIPTextEncoder), as it can embedd text, images and audio to the same latent space.

For more information, such as run executor on gpu, check out [documentation](https://docs.jina.ai/tutorials/gpu-executor/).
