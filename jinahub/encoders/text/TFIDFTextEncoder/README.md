# TFIDFTextEncoder

**TFIDFTextEncoder** is a class that wraps the text embedding functionality of a TFIDF model.

The TFIDF model is a classic vector representation for [information retrieval](https://en.wikipedia.org/wiki/Tf–idf).

`TfidfTextEncoder` encodes data from a `DocumentArray` and updates the `doc.embedding` attributes with a  `scipy.csr_matrix`of floating point values for each doc in DocumentArray.




## Prerequisites



You also need a TF-IDF vectorizer pretrained.

### Pretraining a TF-IDF Vectorizer

The `TFIDFTextEncoder`  uses a `sklearn.feature_extraction.text.TfidfVectorizer`object that needs to be fitted and stored as a pickle object which the `TFIDFTextEncoder` will load from `path_vectorizer`. By default `path_vectorizer='model/tfidf_vectorizer.pickle'` .

The following snipped can be used to fit a `TfidfVectorizer` with a toy corpus. To achieve better performance or adapt the encoder to other languages you can change `load_data` function from below to load any other user specific dataset.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def load_data():
    from sklearn.datasets import fetch_20newsgroups
    newsgroups_train = fetch_20newsgroups(subset='train')
    return newsgroups_train.data

if __name__ == '__main__':
    X = load_data()    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(X)
    pickle.dump(tfidf_vectorizer, open('tfidf_vectorizer.pickle', 'wb'))
```

## Usages

### Via JinaHub

#### using docker images
Use the prebuilt images from JinaHub in your Python code: 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://TFIDFTextEncoder')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://TFIDFTextEncoder'
```

#### using source code
Use the source code from JinaHub in your Python code:

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://TFIDFTextEncoder')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://TFIDFTextEncoder'
```


## Example 

```python
from jina import Flow, Document

f = Flow().add(uses='jinahub+docker://TFIDFTextEncoder')

with f:
    resp = f.index(inputs=Document(text='Han eats pizza'), return_results=True)
	print(f'{resp}')
```

### Inputs

[Documents](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Document.md) with `text`. By default, the input `text`must be a unicode string.  

### Returns

[Documents](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Document.md) with `embedding` fields filled with an `scipy.sparse.csr_matrix` of the shape `n_vocabulary`.



## Reference

https://en.wikipedia.org/wiki/Tf-idf
