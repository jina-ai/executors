import os
import pytest
import pickle

from jina import Executor, requests, Document, DocumentArray
from ..simple_inverted_indexer import SimpleInvertedIndexer
from sklearn.feature_extraction.text import CountVectorizer


@pytest.fixture()
def count_vectorizer_path(tmpdir):
    corpus = ['hello', 'man', 'woman', 'ball', 'ski', 'sport', 'football']
    vectorizer = CountVectorizer(stop_words='english')
    vectorizer.fit(corpus)
    vectorizer_path = os.path.join(str(tmpdir), 'vectorizer.pkl')
    with open(vectorizer_path, 'wb') as fp:
        pickle.dump(vectorizer, fp)
    return vectorizer_path


@pytest.mark.parametrize('relevance_score', ['bm25', 'tfidf'])
def test_simple_inverted_index(relevance_score, count_vectorizer_path, tmpdir):
    doc0 = Document(id='0', text='hello ball outside')
    doc1 = Document(id='1', text='not present word ski')
    doc2 = Document(id='2', text='woman ball football ball')
    da = DocumentArray([doc0, doc1, doc2])

    indexer = SimpleInvertedIndexer(inverted_index_file_name='inverted_index.pkl',
                                    pretrained_count_vectorizer_path=count_vectorizer_path, ranking_method=relevance_score,
                                    metas={'workspace': str(tmpdir)})
    indexer.index(da)
    query = DocumentArray([Document(text='ball ski')])
    indexer.search(query)
    returned_ids = [match.id for match in query[0].matches]
    assert returned_ids == ['1', '0', '2']
