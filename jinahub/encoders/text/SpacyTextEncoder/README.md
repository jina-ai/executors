# SpacyTextEncoder

**SpacyTextEncoder** is a class that encodes text with spaCy models.






## Usages

### Via JinaHub

#### using docker images
Use the prebuilt images from JinaHub in your Python code: 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://SpacyTextEncoder')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://SpacyTextEncoder'
```

#### using source code
Use the source code from JinaHub in your Python code:

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://SpacyTextEncoder')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://SpacyTextEncoder'
```


## Example 


```python
from jina import Flow, Document

f = Flow().add(uses='jinahub+docker://SpacyTextEncoder')

with f:
    resp = f.post(on='foo', inputs=Document(), return_results=True)
	print(f'{resp}')
```


#### Inputs 

`Document` with `text`.

#### Returns

`Document` with `embedding` field filled with spacy vector.

## Reference
- https://spacy.io/models/
- https://spacy.io/usage/models
