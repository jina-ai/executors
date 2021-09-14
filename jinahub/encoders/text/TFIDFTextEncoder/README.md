# TFIDFTextEncoder

**TFIDFTextEncoder** is a class that wraps the text embedding functionality of a TFIDF model.

The TFIDF model is a classic vector representation for [information retrieval](https://en.wikipedia.org/wiki/Tf–idf).

`TfidfTextEncoder` encodes data from a `DocumentArray` and updates the `doc.embedding` attributes with a  `scipy.csr_matrix`of floating point values for each doc in DocumentArray.




## Prerequisites

You need a TF-IDF vectorizer pretrained.

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

## Reference

- https://en.wikipedia.org/wiki/Tf-idf
