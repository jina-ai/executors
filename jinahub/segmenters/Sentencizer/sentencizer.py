__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import re
from string import punctuation
from typing import Dict, List, Optional, Tuple

from jina import Document, DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger
from nltk.tokenize import sent_tokenize


class Sentencizer(Executor):
    """
    :class:`Sentencizer` split the text on the doc-level
    into sentences on the chunk-level with a rule-base strategy.
    The text is split by the punctuation characters listed in ``punct_chars``.
    The sentences that are shorter than the ``min_sent_len``
    or longer than the ``max_sent_len`` after stripping will be discarded.
    """

    def __init__(
        self,
        min_sent_len: int = 1,
        max_sent_len: int = 512,
        punct_chars: Optional[List[str]] = None,
        uniform_weight: bool = True,
        traversal_paths: Tuple[str] = ('r',),
        *args,
        **kwargs
    ):
        """
        :param min_sent_len: the minimal number of characters,
            (including white spaces) of the sentence, by default 1.
        :param max_sent_len: the maximal number of characters,
            (including white spaces) of the sentence, by default 512.
        :param punct_chars: the punctuation characters to split on,
            whatever is in the list will be used,
            for example ['!', '.', '?'] will use '!', '.' and '?'
            If smart_tokenizer=True is passed to segment, punct_chars is
            no longer considered
        :param uniform_weight: the definition of it should have
            uniform weight or should be calculated
        :param traversal_paths: traverse path on docs, e.g. ['r'], ['c']
        """
        super().__init__(*args, **kwargs)
        self.min_sent_len = min_sent_len
        self.max_sent_len = max_sent_len
        self.punct_chars = punct_chars
        self.uniform_weight = uniform_weight
        self.logger = JinaLogger(self.__class__.__name__)
        self.traversal_paths = traversal_paths
        if not punct_chars:
            self.punct_chars = [
                '!',
                '.',
                '?',
                '։',
                '؟',
                '۔',
                '܀',
                '܁',
                '܂',
                '‼',
                '‽',
                '⁇',
                '⁈',
                '⁉',
                '⸮',
                '﹖',
                '﹗',
                '！',
                '．',
                '？',
                '｡',
                '。',
                '\n',
            ]
        if self.min_sent_len > self.max_sent_len:
            self.logger.warning(
                'the min_sent_len (={}) should be smaller or equal to the max_sent_len (={})'.format(
                    self.min_sent_len, self.max_sent_len
                )
            )
        self._slit_pat = re.compile(
            r'\s*([^{0}]+)(?<!\s)[{0}]*'.format(''.join(set(self.punct_chars)))
        )

    def _seg(self, text, **kwargs) -> List:
        ret = [
            (m.group(0), m.start(), m.end()) for m in re.finditer(self._slit_pat, text)
        ]
        return ret

    def _smart_seg(self, text: str, language: str = 'english', **kwargs) -> List:
        """
        Split a string into a list of strings using nltk function sent_tokenize
        Implemented to give a smarter tokenization of abbreviation as Dr. or Mr
        Example:
            - smart_seg: 'Mr. Charles. is sick' -> 'Mr. Charles.', 'is sick'
            - seg: 'Mr. Charles. is sick' -> 'Mr.', 'Charles.', 'is sick'
        Reference: https://www.nltk.org/api/nltk.tokenize.html
        """
        j = 0
        ret = []
        tokenization = sent_tokenize(text)
        for sentence in tokenization:
            if sentence not in punctuation:
                start_idx = j + text[j:].find(sentence[0])
                end_idx = start_idx + len(sentence)
                j += len(sentence)
                ret.append((sentence, start_idx, end_idx))
        return ret

    @requests
    def segment(self, docs: Optional[DocumentArray], parameters: Dict, **kwargs):
        """
        Split the text into sentences.
        :param docs: Documents that contain the text
        :param parameters: Dictionary of parameters
        :param kwargs: Additional keyword arguments
        :return: a list of chunk dicts with the split sentences
        """
        if not docs:
            return
        traversal_path = parameters.get('traversal_paths', self.traversal_paths)
        flat_docs = docs.traverse_flat(traversal_path)
        smart_tokenizer = parameters.get('smart_tokenizer', False)
        language = parameters.get('language', 'english')
        if smart_tokenizer:
            seg_function = self._smart_seg
        else:
            seg_function = self._seg

        for doc in flat_docs:
            text = doc.text
            ret = seg_function(text, language=language)
            if not ret:
                ret = [(text, 0, len(text))]
            for ci, (r, s, e) in enumerate(ret):
                f = re.sub('\n+', ' ', r).strip()
                f = f[: self.max_sent_len]
                if len(f) > self.min_sent_len:
                    doc.chunks.append(
                        Document(
                            text=f,
                            offset=ci,
                            weight=1.0 if self.uniform_weight else len(f) / len(text),
                            location=[s, e],
                        )
                    )
