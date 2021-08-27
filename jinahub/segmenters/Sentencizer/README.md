# Sentencizer

**Sentencizer** is a class that splits texts into sentences.







## Usages

### Via JinaHub

#### using docker images
Use the prebuilt images from JinaHub in your Python code: 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://Sentencizer')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: sentencizer
    uses: 'jinahub+docker://Sentencizer'
```

#### using source code
Use the source code from JinaHub in your Python code:

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://Sentencizer')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: sentencizer
    uses: 'jinahub://Sentencizer'
```

## Example 

```python
from jina import Flow, Document

f = Flow().add(uses='jinahub+docker://Sentencizer')

with f:
    resp = f.post(on='foo', inputs=Document(text='Hello. World.'), return_results=True)
    print(f'{resp}')
```

#### Inputs 

`Document` with `text` containing two sentences split by a dot `.`, namely `Hello. World.`.

#### Returns

`Document` with two `chunks` Documents. The first chunk contains `text='Hello.'`, the second chunk contains `text='World.'`

## Reference
- Used in the multires-lyrics-search example in: https://github.com/jina-ai/examples

