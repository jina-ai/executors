from typing import List

import pytest
import torch
from jina import Document, DocumentArray
from jina.executors import BaseExecutor

from ...dpr_reader import DPRReaderRanker


@pytest.fixture(scope='session')
def basic_ranker() -> DPRReaderRanker:
    return DPRReaderRanker()


@pytest.fixture(scope='function')
def example_docs() -> DocumentArray:
    doc1 = Document(text='What is Jina?')
    matches1 = [
        Document(text='Jina AI is a Neural Search Company', tags={'title': 'Jina AI'}),
        Document(
            text='In Jainism, Jinvani means the message or the teachings of the Jina (arihant).',
            tags={'title': 'Jinvani'},
        ),
        Document(
            text='Lord Mahavir was the twenty-fourth and the last Tirthankara of the Jain religion',
            tags={'title': 'Lord Mahavir and Jain religion'},
        ),
    ]
    doc1.matches.extend(matches1)

    doc2 = Document(text='Is riemann hypothesis solved?')
    matches2 = [
        Document(
            text='Many consider it to be the most important unsolved problem in pure mathematics.',
            tags={'title': 'Riemann Hypothesis'},
        ),
        Document(
            text='“As far as I am concerned, the Riemann Hypothesis remains open,” said Martin Bridson.',
            tags={
                'title': '‘Riemann Hypothesis’ remains open, clarifies math institute'
            },
        ),
    ]
    doc2.matches.extend(matches2)

    docs = DocumentArray([doc1, doc2])
    return docs


def test_config():
    encoder = BaseExecutor.load_config('../../config.yml')
    assert encoder.default_batch_size == 32
    assert encoder.default_traversal_paths == ('r',)
    assert encoder.title_tag_key == 'title'
    assert encoder.num_spans_per_match == 1


def test_no_document(basic_ranker: DPRReaderRanker):
    basic_ranker.rank(None, {})


def test_empty_documents(basic_ranker: DPRReaderRanker):
    docs = DocumentArray([])
    basic_ranker.rank(docs, {})
    assert len(docs) == 0


def test_no_text_documents(basic_ranker: DPRReaderRanker):
    docs = DocumentArray([Document()])
    with pytest.raises(ValueError, match=r'No question \(text\) found for document'):
        basic_ranker.rank(docs, {})


def test_documents_no_matches(basic_ranker: DPRReaderRanker):
    docs = DocumentArray([Document(text='I have no matches')])
    basic_ranker.rank(docs, {})
    assert len(docs) == 1
    assert len(docs[0].matches) == 0


def test_matches_no_title(basic_ranker: DPRReaderRanker):
    doc = Document(text='A question?')
    doc.matches.append(Document(text='I have no titile.'))
    docs = DocumentArray([doc])

    with pytest.raises(ValueError, match='All matches are required to have'):
        basic_ranker.rank(docs, {})


def test_ranking_cpu(basic_ranker: DPRReaderRanker, example_docs: DocumentArray):

    basic_ranker.rank(example_docs, {})

    assert len(example_docs[0].matches) == 3
    assert len(example_docs[1].matches) == 2

    for i in range(3):
        assert 'relevance_score' in example_docs[0].matches[i].scores
        assert 'span_score' in example_docs[0].matches[i].tags
        assert 'title' in example_docs[0].matches[i].tags


def test_spans_title_match(example_docs: DocumentArray):
    ranker = DPRReaderRanker(num_spans_per_match=2)
    texts = {
        'Jina AI': 'Jina AI is a Neural Search Company',
        'Jinvani': 'In Jainism, Jinvani means the message or the teachings of the Jina (arihant).',
        'Lord Mahavir and Jain religion': 'Lord Mahavir was the twenty-fourth and the last Tirthankara of the Jain religion',
    }
    ranker.rank(example_docs, {})

    assert len(example_docs[0].matches) == 6
    for match in example_docs[0].matches:
        assert match.text.lower() in texts[match.tags['title']].lower()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU is needed for this test')
def test_ranking_gpu(example_docs: DocumentArray):

    ranker = DPRReaderRanker(device='gpu')
    ranker.rank(example_docs, {})

    assert len(example_docs[0].matches) == 3
    assert len(example_docs[1].matches) == 2

    for i in range(3):
        assert 'relevance_score' in example_docs[0].matches[i].scores
        assert 'span_score' in example_docs[0].matches[i].tags
        assert 'title' in example_docs[0].matches[i].tags


@pytest.mark.parametrize('batch_size', [1, 2, 4, 8])
def test_batch_size(
    basic_ranker: DPRReaderRanker, example_docs: DocumentArray, batch_size: int
):
    docs = DocumentArray([example_docs[i % 2] for i in range(17)])
    basic_ranker.rank(docs, parameters={'batch_size': batch_size})

    for i, doc in enumerate(docs):
        assert len(doc.matches) == [3, 2][i % 2]

        assert 'relevance_score' in doc.matches[0].scores
        assert 'span_score' in doc.matches[0].tags
        assert 'title' in doc.matches[0].tags


def test_quality_ranking(basic_ranker: DPRReaderRanker, example_docs: DocumentArray):
    """A small test to see that the results make some sense."""
    basic_ranker.rank(example_docs, {})

    assert example_docs[0].matches[0].tags['title'] == 'Jina AI'
    assert example_docs[1].matches[0].text == 'unsolved problem'
    assert example_docs[1].matches[1].text == 'open'
