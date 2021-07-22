__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import List, Dict, Optional

import numpy as np
import torch
import spacy

from jina import Executor, DocumentArray, requests
from jina.logging.logger import JinaLogger


class SpacyTextEncoder(Executor):
    """
    :class:`SpacyTextEncoder` encodes ``Document`` using models offered by Spacy
    :param lang: pre-trained spaCy language pipeline (model name HashEmbedCNN by default for tok2vec), `en_core_web_sm`
        by default. Allows models `en_core_web_md`, `en_core_web_lg`, `en_core_web_trf`. Refer https://spacy.io/models/en.
    :param use_default_encoder: if True will use parser component,
        otherwise tok2vec implementation will be chosen,
        by default False.
    :param default_traversal_paths: fallback traversal path in case there is not traversal path sent in the request
    :param device: device to use for encoding ['cuda', 'cpu] - if not set, the device is detected automatically
    :param args: Additional positional arguments.
    :param kwargs: Additional positional arguments.
    """

    SPACY_COMPONENTS = [
        'tagger',
        'parser',
        'ner',
        'senter',
        'tok2vec',
        'lemmatizer',
        'attribute_ruler',
    ]

    def __init__(self,
                 lang: str = 'en_core_web_sm',
                 use_default_encoder: bool = False,
                 default_traversal_paths: List[str] = ['r'],
                 device: Optional[str] = None,
                 *args, **kwargs):
        """Set constructor."""
        super().__init__(*args, **kwargs)
        self.lang = lang
        self.use_default_encoder = use_default_encoder
        self.default_traversal_paths = default_traversal_paths
        self.logger = JinaLogger(self.__class__.__name__)
        if not device:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        if self.device == 'cuda':
            spacy.require_gpu()

        try:
            self.spacy_model = spacy.load(self.lang)
            # Disable everything as we only requires certain pipelines to turned on.
            ignored_components = []
            for comp in self.SPACY_COMPONENTS:
                try:
                    self.spacy_model.disable_pipe(comp)
                except Exception:
                    ignored_components.append(comp)
            self.logger.info(f'Ignoring {ignored_components} pipelines as it does not available on the model package.')
        except IOError:
            self.logger.error(
                f'spaCy model for language {self.lang} can not be found. Please install by referring to the '
                'official page https://spacy.io/usage/models.'
            )
            raise

        if self.use_default_encoder:
            try:
                self.spacy_model.enable_pipe('parser')
            except ValueError:
                self.logger.error(
                    f'Parser for language {self.lang} can not be found. The default sentence encoder requires'
                    'DependencyParser to be trained. Please refer to https://spacy.io/api/tok2vec for more clarity.'
                )
                raise
        else:
            try:
                self.spacy_model.enable_pipe('tok2vec')
            except ValueError:
                self.logger.error(
                    f'TokenToVector is not available for language {self.lang}. Please refer to'
                    'https://github.com/explosion/spaCy/issues/6615 for training your own recognizer.'
                )
                raise

    @requests
    def encode(self, docs: Optional[DocumentArray], parameters: Dict, **kwargs):
        """
        Encode all docs with text and store the encodings in the embedding attribute of the docs.
        :param docs: documents sent to the encoder. The docs must have `text` as content
        :param parameters: dictionary to define the `traversal_path` and the `batch_size`.
            For example,
            `parameters={'traversal_paths': ['r']}`
            will set the parameters for traversal_paths that is actually used`
        """
        if docs:
            trav_paths = parameters.get('traversal_paths', self.default_traversal_paths)
            # traverse thought all documents which have to be processed
            flat_docs = docs.traverse_flat(trav_paths)
            # filter out documents without text
            filtered_docs = [doc for doc in flat_docs if doc.text is not None]

            for doc in filtered_docs:
                spacy_doc = self.spacy_model(doc.text)
                doc.embedding = spacy_doc.vector
