import os

import scipy

cur_dir = os.path.dirname(os.path.abspath(__file__))


def load_data():
    from sklearn.datasets import fetch_20newsgroups

    newsgroups_train = fetch_20newsgroups(subset='train')
    return newsgroups_train.data


if __name__ == '__main__':
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf_vectorizer = TfidfVectorizer()
    X = load_data()
    tfidf_vectorizer.fit(X)

    # Encoding a single item
    text = ['Han likes eating pizza']
    embedding_array = tfidf_vectorizer.transform(text)
    scipy.sparse.save_npz(os.path.join(cur_dir, 'expected.npz'), embedding_array)

    # Encoding a batch
    text = ['Han likes eating pizza', 'Han likes pizza', 'Jina rocks']
    embedding_batch = tfidf_vectorizer.transform(text)
    scipy.sparse.save_npz(os.path.join(cur_dir, 'expected_batch.npz'), embedding_batch)
