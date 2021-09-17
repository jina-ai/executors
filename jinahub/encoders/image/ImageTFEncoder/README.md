# ImageTFEncoder

`ImageTFEncoder` encodes ``Document`` content from a ndarray, potentially BatchSize x (Height x Width x Channel) into a ndarray of `BatchSize * d`.
Internally, `ImageTFEncoder` wraps the models from [tensorflow.keras.applications](https://keras.io/applications/).

For more information on the `gpu` usage and `volume` mounting, please refer to the [documentation](https://docs.jina.ai/tutorials/gpu-executor/).
