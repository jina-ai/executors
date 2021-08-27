# TextPaddleEncoder

**TextPaddleEncoder** is a class that wraps the text embedding functionality from the **PaddlePaddle** and **PaddleHub**.


`TextPaddleEncoder` encode text stored in the `text` attribute of the [**Document**](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Document.md) and saves the encoding in the embedding attribute.




## Usages

### Via JinaHub

#### Using docker images

Use the prebuilt images from JinaHub in your Python code: 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://TextPaddleEncoder',
               volumes='/your_home_folder/.paddlehub:/root/.paddlehub')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://TextPaddleEncoder'
    volumes: '/your_home_folder/.paddlehub:/root/.paddlehub'
```

#### Using source code

Use the source code from JinaHub in your Python code,

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://TextPaddleEncoder',
			   volumes= '/your_home_folder/.paddlehub:/root/.paddlehub')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://TextPaddleEncoder'
	volumes: '/your_home_folder/.paddlehub:/root/.paddlehub'
```
	


## Example

```python
from jina import Flow, Document

f = Flow().add(uses='jinahub+docker://TextPaddleEncoder',
               volumes='/your_home_folder/.paddlehub:/root/.paddlehub')


def check_resp(resp):
    for doc in resp.data.docs:
        d = Document(doc)
        print(f'embedding shape: {d.embedding.shape}')


with f:
    f.post(on='foo',
           inputs=Document(text='hello world'),
           on_done=check_resp)
	    
```


### Inputs 

[Documents](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Document.md) with `text`.

### Returns

[Documents](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Document.md) with `embedding` fields filled with an `ndarray` of the shape `1024` (depends on the model) with `dtype=nfloat32`.



## Reference
- https://www.paddlepaddle.org.cn/hublist?filter=en_category&value=SemanticModel
