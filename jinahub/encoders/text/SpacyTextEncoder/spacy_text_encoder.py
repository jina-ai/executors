__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import subprocess
from typing import Dict, List, Optional

import spacy
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
        require_gpu: bool = False,
        download_data: bool = True,
        default_batch_size: int = 32,
        default_traversal_paths: List[str] = ['r'],
        *args,
        **kwargs,
    ):
        """
        :param model_name: pre-trained spaCy language pipeline name
        :param require_gpu: device to use for encoding ['cuda', 'cpu] - if not set,
            the device is detected automatically
        :param default_batch_size: Default batch size, used if ``batch_size`` is not
            provided as a parameter in the request
        :param default_traversal_paths: Default traversal paths, used if ``traversal_paths``
            are not provided as a parameter in the request.
        :param args: Additional positional arguments.
        :param kwargs: Additional positional arguments.
        """
        super().__init__(*args, **kwargs)

        self.default_batch_size = default_batch_size
        self.default_traversal_paths = default_traversal_paths

        if require_gpu:
            spacy.require_gpu()

        if download_data:
            subprocess.run(
                ['python', '-m', 'spacy', 'download', model_name], check=True
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
                    doc.embedding = spacy_doc.vector
