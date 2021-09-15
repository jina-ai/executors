# VggishAudioEncoder

**VggishAudioEncoder** is a class that wraps the [VGGISH](https://github.com/tensorflow/models/tree/master/research/audioset/vggish) model for generating embeddings for audio data. 



#### GPU usage

You can use the GPU via the source code. Therefore, you need a matching CUDA version
and GPU drivers installed on your system. 

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://VGGISHAudioEncoder'
    uses_with:
      device: 'cuda'
```

Alternatively, use the GPU docker container. Therefore, you need GPU
drivers installed on your system and nvidia-docker installed.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://VGGISHAudioEncoder/gpu'
    gpus: all
    uses_with:
      device: 'cuda'
```


## Reference
- [VGGISH paper](https://research.google/pubs/pub45611/)
- [VGGISH code](https://github.com/tensorflow/models/tree/master/research/audioset/vggish)
