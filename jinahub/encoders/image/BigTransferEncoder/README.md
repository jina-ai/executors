# Big Transfer Image Encoder

**Big Transfer Image Encoder** is a class that uses the Big Transfer models presented by Google [here]((https://github.com/google-research/big_transfer)).
It uses a pretrained version of a BiT model to encode an image from an array of shape 
(Batch x (Channel x Height x Width)) into an array of shape (Batch x Encoding) 


#### GPU usage

You can use the GPU via the source code. Therefore, you need a matching CUDA version
and GPU drivers installed on your system. 

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://BigTransferEncoder'
    uses_with:
      device: '/GPU:0'
```

Alternatively, use the GPU docker container. Therefore, you need GPU
drivers installed on your system and nvidia-docker installed.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://BigTransferEncoder/gpu'
    gpus: all
    uses_with:
      device: '/GPU:0'
```


## Reference
- https://github.com/google-research/big_transfer
- https://tfhub.dev/google/collections/bit/1
