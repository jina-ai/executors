# ImageTFEncoder

`ImageTFEncoder` encode the ``blob`` with the size (Height x Width x Channel) of ``Document`` into one dimensional vector and stored it in the ``embedding`` field
Internally, `ImageTFEncoder` wraps the models from [tensorflow.keras.applications](https://keras.io/applications/).

For more information on the `gpu` usage and `volume` mounting, please refer to the [documentation](https://docs.jina.ai/tutorials/gpu-executor/).
