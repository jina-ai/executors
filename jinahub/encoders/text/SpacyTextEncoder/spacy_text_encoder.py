__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import subprocess
from typing import Dict, Iterable, Optional

import spacy
from cupy import asnumpy
from jina import DocumentArray, Executor, requests
from jina_commons.batching import get_docs_batch_generator

_EXCLUDE_COMPONENTS = [
    'tagger',
    'parser',
    'ner',
    'senter',
    'lemmatizer',
    'attribute_ruler',
]


class SpacyTextEncoder(Executor):
    """
    :class:`SpacyTextEncoder` encodes ``Document`` using models offered by Spacy
    """

    def __init__(
        self,
        model_name: str = 'en_core_web_sm',
        download_data: bool = True,
        default_batch_size: int = 32,
        default_traversal_paths: Iterable[str] = ('r',),
        device: str = 'cpu',
        *args,
        **kwargs,
    ):
        """
        :param model_name: pre-trained spaCy language pipeline name
        :param default_batch_size: fallback batch size in case there is not batch size sent in the request
        :param default_traversal_paths: fallback traversal path in case there is not traversal path sent in the request
        :param device: device to use for encoding ['cuda', 'cpu', 'cuda:2']
        """
        super().__init__(*args, **kwargs)

        self.default_batch_size = default_batch_size
        self.default_traversal_paths = default_traversal_paths
        if device.startswith('cuda'):
            spacy.require_gpu()
        if download_data:
            subprocess.run(
                ['python3', '-m', 'spacy', 'download', model_name], check=True
            )
        self.spacy_model = spacy.load(model_name, exclude=_EXCLUDE_COMPONENTS)

    @requests
    def encode(self, docs: Optional[DocumentArray], parameters: Dict, **kwargs):
        """
        Encode all docs with text and store the encodings in the embedding
        attribute of the docs.

        :param docs: documents sent to the encoder. The docs must have the
            ``text`` attribute.
        :param parameters: dictionary to define the ``traversal_path`` and the
            ``batch_size``. For example,
            ``parameters={'traversal_paths': ['r'], 'batch_size': 10}``
        """
        if docs:
            batch_size = parameters.get('batch_size', self.default_batch_size)
            document_batches_generator = get_docs_batch_generator(
                docs,
                traversal_path=parameters.get(
                    'traversal_paths', self.default_traversal_paths
                ),
                batch_size=batch_size,
                needs_attr='text',
            )
            for document_batch in document_batches_generator:
                texts = [doc.text for doc in document_batch]
                for doc, spacy_doc in zip(
                    document_batch, self.spacy_model.pipe(texts, batch_size=batch_size)
                ):
                    if self.device.startswith('cuda'):
                        doc.embedding = asnumpy(spacy_doc.vector)
                    else:
                        doc.embedding = spacy_doc.vector
