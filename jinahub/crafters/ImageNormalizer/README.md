# Image Normalizer

**Image Normalizer** is a class that resizes, crops and normalizes images.
Since normalization is highly dependent on the model, 
it is recommended to have it as part of the encoders instead of using it in a dedicated executor.
Therefore, misconfigurations resulting in a training-serving-gap are less likely.