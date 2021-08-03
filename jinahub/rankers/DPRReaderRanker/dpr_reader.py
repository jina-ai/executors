from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch

from jina import Document, DocumentArray, Executor, requests
from transformers import DPRReader, DPRReaderTokenizerFast


def _batcher(iterable, n=1):
    """Batch an iterable. Here temporarily until
    https://github.com/jina-ai/jina/issues/3068
    is solved.
    """
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : ndx + n]


def _sort_by(x: List, sort_by: List, reverse: bool = True) -> List:
    """Sort a list x by another list (sort_by)"""
    return [elem_x for (_, elem_x) in sorted(zip(sort_by, x), reverse=reverse)]


def _logistic_fn(x: np.ndarray) -> List[float]:
    """Compute the logistic function"""
    return (np.exp(x) / (1 + np.exp(x))).tolist()


class DPRReaderRanker(Executor):
    """
    This executor first extracts answers (answers spans) from all the matches,
    ranks them according to their relevance score, and then replaces the original
    matches with these extracted answers.

    This executor uses the DPR Reader model to re-rank documents based on
    cross-attention between the question (main document text) and the answer
    passages (text of the matches + their titles, if specified).

    :param pretrained_model_name_or_path: Can be either:
        - the model id of a pretrained model hosted inside a model repo
          on huggingface.co.
        - A path to a directory containing model weights, saved using
          the transformers model's ``save_pretrained()`` method
    :param base_tokenizer_model: Base tokenizer model. The possible values are
        the same as for the ``pretrained_model_name_or_path`` parameters. If not
        provided, the ``pretrained_model_name_or_path`` parameter value will be used
    :param title_tag_key: The key of the tag that contains document title in the
        match documents. Specify it if you want the text of the matches to be combined
        with their titles (to mirror the method used in training of the original model)
    :param num_spans_per_match: Number of spans to extract per match
    :param max_length: Max length argument for the tokenizer
    :param default_batch_size: Default batch size for processing documents, used if the
        batch size is not passed as a parameter with the request.
    :param default_traversal_paths: Default traversal paths for processing documents,
        used if the traversal path is not passed as a parameter with the request.
    :param device: The device (cpu or gpu) that the model should be on.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = 'facebook/dpr-reader-single-nq-base',
        base_tokenizer_model: Optional[str] = None,
        title_tag_key: Optional[str] = None,
        num_spans_per_match: int = 1,
        max_length: Optional[int] = None,
        default_batch_size: int = 32,
        default_traversal_paths: Iterable[str] = ('r',),
        device: str = 'cpu',
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.title_tag_key = title_tag_key
        self.device = device
        self.max_length = max_length
        self.num_spans_per_match = num_spans_per_match

        if not base_tokenizer_model:
            base_tokenizer_model = pretrained_model_name_or_path

        self.tokenizer = DPRReaderTokenizerFast.from_pretrained(base_tokenizer_model)
        self.model = DPRReader.from_pretrained(pretrained_model_name_or_path)

        self.model = self.model.to(torch.device(self.device)).eval()

        self.default_traversal_paths = default_traversal_paths
        self.default_batch_size = default_batch_size

    @requests
    def rank(
        self, docs: Optional[DocumentArray], parameters: dict, **kwargs
    ) -> DocumentArray:
        """
        Extracts answers from existing matches, (re)ranks them, and replaces the current
        matches with extracted answers.

        For each match ``num_spans_per_match`` of answers will be extracted, which
        means that the new matches of the document will have a length of previous
        number of matches times ``num_spans_per_match``.

        The new matches will be have a score called ``relevance_score`` saved under
        their scores. They will also have a tag ``span_score``, which refers to their
        span score, which is used to rank answers that come from the same match.

        If you specified ``title_tag_key`` at initialization, the tag ``title`` will
        also be added to the new matches, and will equal the title of the match from
        which they were extracted.

        :param docs: Documents whose matches to re-rank (specifically, the matches of
            the documents on the traversal paths will be re-ranked). The document's
            ``text`` attribute is taken as the question, and the ``text`` attribute
            of the matches as the context. If you specified ``title_tag_key`` at
            initialization, the matches must also have a title (under this tag).
        :param parameters: dictionary to define the ``traversal_path`` and the
            ``batch_size``. For example::
            parameters={'traversal_paths': ['r'], 'batch_size': 10}
        """

        if not docs:
            return None

        for doc in docs.traverse_flat(
            parameters.get('traversal_paths', self.default_traversal_paths)
        ):
            if not doc.text:
                raise ValueError(
                    f'No question (text) found for document with id {doc.id}.'
                    ' DPRReaderRanker requires a question from the main document'
                    ' and context (text) from its matches.'
                )

            answer_spans = []
            answer_relevance_scores = []
            answer_span_scores = []
            answer_titles = []

            match_batches_generator = _batcher(
                doc.matches, n=parameters.get('batch_size', self.default_batch_size)
            )
            for matches in match_batches_generator:
                contexts = matches.get_attributes('text')

                titles = None
                if self.title_tag_key:
                    titles = matches.get_attributes(f'tags__{self.title_tag_key}')

                    if len(titles) != len(matches):
                        raise ValueError(
                            'All matches are required to have the'
                            f' {self.title_tag_key} tag, but found'
                            f' {len(titles) - len(matches)} matches without it.'
                        )

                with torch.no_grad():
                    (
                        relevance_scores,
                        span_scores,
                        spans,
                        titles,
                    ) = self._get_outputs(doc.text, contexts, titles)

                answer_spans += spans
                answer_relevance_scores += relevance_scores
                answer_span_scores += span_scores
                answer_titles += titles

            # Make sure answers are sorted by relevance scores
            sorting_list = list(zip(answer_relevance_scores, answer_span_scores))
            answer_spans = _sort_by(answer_spans, sorting_list)
            answer_relevance_scores = _sort_by(answer_relevance_scores, sorting_list)
            answer_span_scores = _sort_by(answer_span_scores, sorting_list)
            answer_titles = _sort_by(answer_titles, sorting_list)

            # Replace previous matches with actual answers
            doc.matches.clear()
            for span, rel_score, span_score, title in zip(
                answer_spans, answer_relevance_scores, answer_span_scores, answer_titles
            ):
                scores = {'relevance_score': rel_score}
                tags = {'span_score': span_score}
                if title:
                    tags['title'] = title
                doc.matches.append(Document(text=span, scores=scores, tags=tags))

    def _get_outputs(
        self, question: str, contexts: List[str], titles: Optional[List[str]]
    ) -> Tuple[List[float], List[float], List[str], List[Optional[str]]]:

        encoded_inputs = self.tokenizer(
            questions=question,
            titles=titles,
            texts=contexts,
            padding='longest',
            return_tensors='pt',
        )
        outputs = self.model(**encoded_inputs)

        # For each context, extract num_spans_per_match best spans
        best_spans = self.tokenizer.decode_best_spans(
            encoded_inputs, outputs, num_spans_per_passage=self.num_spans_per_match
        )
        relevance_scores = _logistic_fn([span.relevance_score for span in best_spans])
        span_scores = _logistic_fn([span.span_score for span in best_spans])
        spans = [span.text for span in best_spans]
        if titles:
            answer_titles = [titles[span.doc_id] for span in best_spans]
        else:
            answer_titles = [None] * len(best_spans)

        return relevance_scores, span_scores, spans, answer_titles
