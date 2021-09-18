from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from jina import Document, DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger
from jina_commons.batching import get_docs_batch_generator
from transformers import DPRReader, DPRReaderTokenizerFast


def _logistic_fn(x: np.ndarray) -> List[float]:
    """Compute the logistic function"""
    return (1 / (1 + np.exp(-x))).tolist()


class DPRReaderRanker(Executor):
    """
    This executor first extracts answers (answers spans) from all the matches,
    ranks them according to their relevance score, and then replaces the original
    matches with these extracted answers.

    This executor uses the DPR Reader model to re-rank documents based on
    cross-attention between the question (main document text) and the answer
    passages (text of the matches + their titles, if specified).
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = 'facebook/dpr-reader-single-nq-base',
        base_tokenizer_model: Optional[str] = None,
        title_tag_key: Optional[str] = None,
        num_spans_per_match: int = 2,
        max_length: Optional[int] = None,
        traversal_paths: Iterable[str] = ('r',),
        batch_size: int = 32,
        device: str = 'cpu',
        *args,
        **kwargs,
    ):
        """
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
        :param traversal_paths: Default traversal paths for processing documents,
            used if the traversal path is not passed as a parameter with the request.
        :param batch_size: Default batch size for processing documents, used if the
            batch size is not passed as a parameter with the request.
        :param device: The device (cpu or gpu) that the model should be on.
        """
        super().__init__(*args, **kwargs)
        self.title_tag_key = title_tag_key
        self.device = device
        self.max_length = max_length
        self.num_spans_per_match = num_spans_per_match
        self.logger = JinaLogger(self.__class__.__name__)

        if not base_tokenizer_model:
            base_tokenizer_model = pretrained_model_name_or_path

        self.tokenizer = DPRReaderTokenizerFast.from_pretrained(base_tokenizer_model)
        self.model = DPRReader.from_pretrained(pretrained_model_name_or_path)

        self.model = self.model.to(torch.device(self.device)).eval()

        self.traversal_paths = traversal_paths
        self.batch_size = batch_size

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
            parameters.get('traversal_paths', self.traversal_paths)
        ):
            if not doc.text:
                self.logger.warning(
                    f'No question (text) found for document with id {doc.id}; skipping'
                    ' document. DPRReaderRanker requires a question from the main'
                    ' document and context (text) from its matches.'
                )
                continue

            new_matches = []

            match_batches_generator = get_docs_batch_generator(
                DocumentArray([doc]),
                traversal_path=['m'],
                batch_size=parameters.get('batch_size', self.batch_size),
                needs_attr='text',
            )
            for matches in match_batches_generator:
                inputs = self._prepare_inputs(doc, matches)
                with torch.no_grad():
                    new_matches += self._get_new_matches(*inputs)

            # Make sure answers are sorted by relevance scores
            new_matches.sort(
                key=lambda x: (
                    x.scores['relevance_score'].value,
                    x.scores['span_score'].value,
                ),
                reverse=True,
            )

            # Replace previous matches with actual answers
            doc.matches = new_matches

    def _prepare_inputs(
        self, doc: Document, matches: DocumentArray
    ) -> Tuple[str, List[str], Optional[List[str]]]:
        question = doc.text
        contexts = matches.get_attributes('text')

        titles = None
        if self.title_tag_key:
            titles = matches.get_attributes(f'tags__{self.title_tag_key}')

            if len(titles) != len(matches) or None in titles:
                raise ValueError(
                    f'All matches are required to have the {self.title_tag_key}'
                    ' tag, but found some matches without it.'
                )

        return question, contexts, titles

    def _get_new_matches(
        self, question: str, contexts: List[str], titles: Optional[List[str]]
    ) -> List[Document]:

        encoded_inputs = self.tokenizer(
            questions=[question] * len(contexts),
            titles=titles,
            texts=contexts,
            padding='longest',
            return_tensors='pt',
        ).to(self.device)
        outputs = self.model(**encoded_inputs)

        # For each context, extract num_spans_per_match best spans
        best_spans = self.tokenizer.decode_best_spans(
            encoded_inputs,
            outputs,
            num_spans=self.num_spans_per_match * len(contexts),
            num_spans_per_passage=self.num_spans_per_match,
        )

        new_matches = []
        for span in best_spans:
            new_match = Document(text=span.text)
            new_match.scores['relevance_score'] = _logistic_fn(
                span.relevance_score.cpu()
            )
            new_match.scores['span_score'] = _logistic_fn(span.span_score.cpu())
            if titles:
                new_match.tags['title'] = titles[span.doc_id]
            new_matches.append(new_match)

        return new_matches
