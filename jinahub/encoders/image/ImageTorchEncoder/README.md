# ImageTorchEncoder

**ImageTorchEncoder** wraps the models from [torchvision](https://pytorch.org/vision/stable/index.html).

**ImageTorchEncoder** encodes `Document` blobs of type a `ndarray` and shape Batch x Height x Width x Channel 
into a `ndarray` of Batch x Dim and stores them in the `embedding` attribute of the `Document`.


## Reference

- [PyTorch TorchVision Transformers Preprocessing](https://sparrow.dev/torchvision-transforms/)
