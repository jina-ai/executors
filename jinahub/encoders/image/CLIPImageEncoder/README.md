# CLIPImageEncoder

 **CLIPImageEncoder** is a class that wraps the image embedding functionality from the **CLIP** model.

The **CLIP** model originally was proposed in [Learning Transferable Visual Models From Natural Language Supervision](https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language_Supervision.pdf).

`CLIPImageEncoder` encode images stored in the blob attribute of the [**Document**](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Document.md) and saves the encoding in the embedding attribute.



## Prerequisites

None


## Usages

### Via JinaHub

Use the prebuilt images from JinaHub in your python codes, 

```python
from jina import Flow
	
f = Flow().add(
        uses='jinahub+docker://CLIPImageEncoder',
        volumes='/your_home_folder/.cache/clip:/root/.cache/clip')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://CLIPImageEncoder'
    volumes: '/your_home_folder/.cache/clip:/root/.cache/clip'
```


### Via Pypi

1. Install the `jinahub-clip-image`

	```bash
	pip install git+https://github.com/jina-ai/executor-clip-image.git
	```

1. Use `jinahub-clip-image` in your code

	```python
	from jinahub.encoder.clip_image import CLIPImageEncoder
	from jina import Flow
	
	f = Flow().add(uses=CLIPImageEncoder)
	```


### Via Docker

1. Clone the repo and build the docker image

	```shell
	git clone https://github.com/jina-ai/executor-clip-image.git
	cd executor-clip-image
	docker build -t jinahub-clip-image .
	```

1. Use `jinahub-clip-image` in your codes

	```python
	from jina import Flow
	
	f = Flow().add(
	        uses='docker://jinahub-clip-image:latest',
	        volumes='/your_home_folder/.cache/clip:/root/.cache/clip')
	```
	


## Example 


```python
f = Flow().add(uses='jinahub+docker://CLIPImageEncoder',
               volumes='/Users/nanwang/.cache/clip:/root/.cache/clip')


def check_resp(resp):
    for _doc in resp.data.docs:
        doc = Document(_doc)
        print(f'embedding shape: {doc.embedding.shape}')


with f:
    f.post(on='foo',
           inputs=Document(blob=np.ones((800, 224, 3), dtype=np.uint8)),
           on_done=check_resp)
	    
```


### Inputs 

[Documents](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Document.md) with `blob` of the shape `Height x Width x 3`. By default, the input `blob` must be an `ndarray` with `dtype=uint8`. The `Height` and `Width` can have arbitrary values. When setting `use_default_preprocessing=False`, the input `blob` must have the size of `224x224x3` with `dtype=float32`.

### Returns

[Documents](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Document.md) with `embedding` fields filled with an `ndarray` of the shape `512` with `dtype=nfloat32`.



## Reference
- https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language_Supervision.pdf
- https://github.com/openai/CLIP

