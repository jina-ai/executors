from jina import Executor, requests, Document, DocumentArray
from ...simple_inverted_indexer import SimpleInvertedIndexer


def test_simple_inverted_index(tmpdir):
    doc0 = Document(text='hello ball outside')
    doc1 = Document(text='not present word ski')
    doc2 = Document(text='she ball football ball')
    da = DocumentArray([doc0, doc1, doc2])

    indexer = SimpleInvertedIndexer(inverted_index_file_name='inverted_index.pkl', metas={'workspace': str(tmpdir)})
    indexer.index(da)
    indexer.cache_idfs()
    query = DocumentArray([Document(text='ball ski')])
    indexer.search(query)
    print(f' query {query[0].matches}')
