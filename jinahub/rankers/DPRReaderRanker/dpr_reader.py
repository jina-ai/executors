from itertools import groupby
from typing import Dict, Iterable, List, Optional, Tuple, Union

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
        num_spans_per_match: int = 2,
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

        new_matches = []
        old_matches = []

        docs_list = []
        old_match_ref_doc_dict = {}
        new_match_ref_doc_dict = {}

        # Perpare a list of docs to add matches to later and extract their
        # matches, plus build a reference to their reference docs
        trav_paths = parameters.get('traversal_paths', self.default_traversal_paths)
        for ind, doc in enumerate(docs.traverse_flat(trav_paths)):
            if not doc.text:
                raise ValueError(
                    f'No question (text) found for document with id {doc.id}.'
                    ' DPRReaderRanker requires a question from the main document'
                    ' and context (text) from its matches.'
                )

            docs_list.append(doc)
            old_matches += doc.matches
            for match in doc.matches:
                old_match_ref_doc_dict[match.id] = ind

        # Traverse all matches
        batch_size = parameters.get('batch_size', self.default_batch_size)
        for matches in _batcher(old_matches, n=batch_size):

            inputs = self._prepare_inputs(matches, docs_list, old_match_ref_doc_dict)
            with torch.no_grad():
                outputs = self._get_outputs(*inputs)

            for rel_score, span, tags, ref_ind in zip(*outputs):
                scores = {'relevance_score': rel_score}
                new_match = Document(text=span, scores=scores, tags=tags)
                new_matches.append(new_match)
                new_match_ref_doc_dict[new_match.id] = ref_ind

        # Clear old matches
        for doc in docs_list:
            doc.matches.clear()

        # Append new matches, first sorting them
        for ref_ind, new_matches_group in groupby(
            new_matches, lambda x: new_match_ref_doc_dict[x.id]
        ):
            new_matches_list = list(new_matches_group)
            new_matches_list.sort(
                key=lambda x: (x.scores['relevance_score'].value, x.tags['span_score']),
                reverse=True,
            )

            docs_list[ref_ind].matches.extend(new_matches_list)

    def _prepare_inputs(
        self,
        matches: List[Document],
        docs_list: Dict[str, Document],
        old_match_ref_doc_dict: Dict[str, str],
    ) -> Tuple[List[str], List[str], Optional[List[str]], List[str]]:
        contexts = []
        questions = []
        ref_inds = []
        for match in matches:
            contexts.append(match.text)
            ref_inds.append(old_match_ref_doc_dict[match.id])
            questions.append(docs_list[old_match_ref_doc_dict[match.id]].text)

        titles = None
        if self.title_tag_key:
            try:
                titles = [match.tags[self.title_tag_key] for match in matches]
            except KeyError:
                raise ValueError(
                    f'All matches are required to have the {self.title_tag_key} tag,'
                    ' but found somem atches without it.'
                )

        return questions, contexts, titles, ref_inds

    def _get_outputs(
        self,
        questions: List[str],
        contexts: List[str],
        titles: Optional[List[str]],
        ref_inds: List[str],
    ) -> Tuple[List[float], List[str], List[Dict[str, Union[str, float]]], List[str]]:

        encoded_inputs = self.tokenizer(
            questions=questions,
            titles=titles,
            texts=contexts,
            padding='longest',
            return_tensors='pt',
        )
        outputs = self.model(**encoded_inputs)

        # For each context, extract num_spans_per_match best spans
        best_spans = self.tokenizer.decode_best_spans(
            encoded_inputs,
            outputs,
            num_spans=self.num_spans_per_match * len(questions),
            num_spans_per_passage=self.num_spans_per_match,
        )
        relevance_scores = _logistic_fn([span.relevance_score for span in best_spans])
        spans = [span.text for span in best_spans]
        answer_ref_inds = [ref_inds[span.doc_id] for span in best_spans]

        tags = []
        for span in best_spans:
            tags_span = {'span_score': _logistic_fn(span.span_score)}
            if titles:
                tags_span['title'] = titles[span.doc_id]
            tags.append(tags_span)

        return relevance_scores, spans, tags, answer_ref_inds
