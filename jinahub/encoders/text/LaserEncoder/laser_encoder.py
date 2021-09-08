__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Iterable, List, Optional

import torch
from jina import DocumentArray, Executor, requests
from jina_commons.batching import get_docs_batch_generator
from laserembeddings import Laser


class LaserEncoder(Executor):
    """
    Encode an array of string in size `B` into an ndarray in size `B x D`

    The ndarray potentially is BatchSize x (Channel x Height x Width)

    :class:`LaserEncoder` is a encoder based on Facebook Research's LASER
    (Language-Agnostic SEntence Representations) to compute multilingual
    sentence embeddings: https://github.com/facebookresearch/LASER
    :param path_to_bpe_codes: path to bpe codes from Laser.
        Defaults to Laser.DEFAULT_BPE_CODES_FILE.
    :param path_to_bpe_vocab: path to bpe vocabs from Laser.
        Defaults to Laser.DEFAULT_BPE_VOCAB_FILE.
    :param path_to_encoder: path to the encoder from Laser.
        Defaults to Laser.DEFAULT_ENCODER_FILE.
    :param default_batch_size: size of each batch
    :param default_traversal_paths: traversal path of the Documents, (e.g. 'r', 'c')
    :param device: Device string ('cpu'/'cuda'/'cuda:2')
    :param language: language of the text. Defaults to english(en).
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
    """

    def __init__(
        self,
        path_to_bpe_codes: Optional[str] = None,
        path_to_bpe_vocab: Optional[str] = None,
        path_to_encoder: Optional[str] = None,
        device: str = 'cpu',
        default_batch_size: int = 32,
        default_traversal_paths: Optional[List[str]] = None,
        language: str = 'en',
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._path_to_bpe_codes = path_to_bpe_codes
        self._path_to_bpe_vocab = path_to_bpe_vocab
        self._path_to_encoder = path_to_encoder
        self.device = device
        self.default_batch_size = default_batch_size
        self.default_traversal_paths = default_traversal_paths or ['r']
        self.language = language.lower()

        self.model = Laser(
            bpe_codes=self._path_to_bpe_codes,
            bpe_vocab=self._path_to_bpe_vocab,
            encoder=self._path_to_encoder,
        )
        self.device = torch.device(device)
        self.model.bpeSentenceEmbedding.encoder.encoder.to(self.device)

    @requests
    def encode(self, docs: Optional[DocumentArray], parameters: dict, **kwargs):
        """
        Encode all docs with text and store the encodings in the embedding attribute of the docs.

        :param docs: documents sent to the encoder. The docs must have text.
        :param parameters: dictionary to define the `traversal_path` and the `batch_size`.
            For example,
            `parameters={'traversal_paths': ['r'], 'batch_size': 10}`
            will set the parameters for traversal_paths, batch_size and that are actually used
        """
        if docs:
            document_batches_generator = get_docs_batch_generator(
                docs,
                traversal_path=parameters.get(
                    'traversal_paths', self.default_traversal_paths
                ),
                batch_size=parameters.get('batch_size', self.default_batch_size),
                needs_attr='text',
            )
            self._create_embeddings(document_batches_generator)

    def _create_embeddings(self, document_batches_generator: Iterable):
        for document_batch in document_batches_generator:
            text_batch = [d.text for d in document_batch]

            embeddings = self.model.embed_sentences(text_batch, lang=self.language)
            for document, embedding in zip(document_batch, embeddings):
                document.embedding = embedding
