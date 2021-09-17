# ImageTFEncoder

`ImageTFEncoder` encodes ``Document`` content from a ndarray, potentially BatchSize x (Height x Width x Channel) into a ndarray of `BatchSize * d`. Internally, :class:`ImageTFEncoder` wraps the models from `tensorflow.keras.applications`. https://keras.io/applications/