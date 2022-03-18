__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import subprocess
from typing import Dict, Optional

import spacy
from docarray import DocumentArray
from jina import Executor, requests

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
        traversal_paths: str = '@r',
        batch_size: int = 32,
        device: str = 'cpu',
        *args,
        **kwargs,
    ):
        """
        :param model_name: pre-trained spaCy language pipeline name
        :param traversal_paths: fallback traversal path in case there is not traversal path sent in the request
        :param batch_size: fallback batch size in case there is not batch size sent in the request
        :param device: device to use for encoding.  ['cuda', 'cpu', 'cuda:2']
        """
        super().__init__(*args, **kwargs)

        self.batch_size = batch_size
        self.traversal_paths = traversal_paths
        self.device = device
        if device.startswith('cuda'):
            spacy.require_gpu()
        if download_data:
            subprocess.run(
                ['python3', '-m', 'spacy', 'download', model_name], check=True
            )
        self.spacy_model = spacy.load(model_name, exclude=_EXCLUDE_COMPONENTS)

    @requests
    def encode(
        self, docs: Optional[DocumentArray] = None, parameters: Dict = {}, **kwargs
    ):
        """
        Encode all docs with text and store the encodings in the embedding
        attribute of the docs.

        :param docs: documents sent to the encoder. The docs must have the
            ``text`` attribute.
        :param parameters: dictionary to define the ``traversal_path`` and the
            ``batch_size``. For example,
            ``parameters={'traversal_paths': '@r', 'batch_size': 10}``
        """
        if self.device.startswith('cuda'):
            from cupy import asnumpy

        if docs:
            trav_path = parameters.get('traversal_paths', self.traversal_paths)

            batch_size = parameters.get('batch_size', self.batch_size)

            docs_batch_generator = DocumentArray(
                filter(
                    lambda x: bool(x.text),
                    docs[trav_path],
                )
            ).batch(batch_size=parameters.get('batch_size', self.batch_size))

            for document_batch in docs_batch_generator:
                texts = [doc.text for doc in document_batch]
                for doc, spacy_doc in zip(
                    document_batch, self.spacy_model.pipe(texts, batch_size=batch_size)
                ):
                    if self.device.startswith('cuda'):
                        doc.embedding = asnumpy(spacy_doc.vector)
                    else:
                        doc.embedding = spacy_doc.vector
