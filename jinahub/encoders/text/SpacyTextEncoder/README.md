# SpacyTextEncoder

**SpacyTextEncoder** is a class that encodes text with spaCy models.

#### Inputs 

`Document` with `text`.

#### Returns

`Document` with `embedding` field filled with spacy vector.

## Usage 


```python
from jina import Flow, Document

f = Flow().add(uses='jinahub+docker://SpacyTextEncoder')

with f:
    resp = f.post(on='foo', inputs=Document(), return_results=True)
	print(f'{resp}')
```

## Reference
- https://spacy.io/models/
- https://spacy.io/usage/models
