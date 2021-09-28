__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"


import subprocess
from typing import Iterable, Optional

import torch
from jina import DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger
from laserembeddings import Laser


class LaserEncoder(Executor):
    """
    LaserEncoder is a text encoder based on Facebook Research's LASER encoder.

    :class:`LaserEncoder` is a encoder based on Facebook Research's LASER
    (Language-Agnostic SEntence Representations) to compute multilingual
    sentence embeddings: https://github.com/facebookresearch/LASER
    This encoder is suitable for producing multi-lingual sentence embeddings, enabling
    you to have sentences from multiple languages in the same latent space.
    """

    def __init__(
        self,
        path_to_bpe_codes: Optional[str] = None,
        path_to_bpe_vocab: Optional[str] = None,
        path_to_encoder: Optional[str] = None,
        download_data: bool = True,
        language: str = 'en',
        traversal_paths: Iterable[str] = ('r',),
        batch_size: int = 32,
        device: str = 'cpu',
        *args,
        **kwargs,
    ):
        """
        :param path_to_bpe_codes: path to bpe codes from Laser. Defaults to
            ``Laser.DEFAULT_BPE_CODES_FILE.``
        :param path_to_bpe_vocab: path to bpe vocabs from Laser. Defaults to
            ``Laser.DEFAULT_BPE_VOCAB_FILE``.
        :param path_to_encoder: path to the encoder from Laser. Defaults to
            ``Laser.DEFAULT_ENCODER_FILE``.
        :param download_data: Whether data should be downloaded on initialization. This is
            convenient when just trying out the encoder, but should be turned off in a
            production setting (where you should already have the data on disk), as it can
            lead to large startup times.
        :param language: The default language of the text. Can be overriden by a
            request parameter. The full list of possible values can be found at
            [LASER](https://github.com/facebookresearch/LASER#supported-languages)
            with the language code
            ([ISO 639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes))
        :param traversal_paths: traversal path of the Documents, (e.g. 'r', 'c')
        :param batch_size: size of each batch
        :param device: Device string ('cpu'/'cuda'/'cuda:2')
        """
        super().__init__(*args, **kwargs)
        self.logger = JinaLogger(self.__class__.__name__)

        self._path_to_bpe_codes = path_to_bpe_codes
        self._path_to_bpe_vocab = path_to_bpe_vocab
        self._path_to_encoder = path_to_encoder
        self.device = device
        self.batch_size = batch_size
        self.traversal_paths = traversal_paths
        self.language = language

        if download_data:
            self.logger.info("Downloading data for the Laser model")
            subprocess.run(
                ['python', '-m', 'laserembeddings', 'download-models'], check=True
            )

        self.model = Laser(
            bpe_codes=self._path_to_bpe_codes,
            bpe_vocab=self._path_to_bpe_vocab,
            encoder=self._path_to_encoder,
            embedding_options={'cpu': self.device == 'cpu'},
        )
        self.device = torch.device(device)
        self.model.bpeSentenceEmbedding.encoder.encoder.to(self.device)

    @requests
    def encode(
        self, docs: Optional[DocumentArray] = None, parameters: dict = {}, **kwargs
    ):
        """
        Encode all docs with text and store the encodings in the embedding attribute
        of the docs.

        :param docs: documents sent to the encoder. The docs must have the ``text``
            attribute.
        :param parameters: dictionary to define the ``traversal_path``, the
            ``batch_size`` and ``language``. For example,
            ``{'traversal_paths': ['r'], 'batch_size': 10}``. This will override the
            default parameters set at init.
        """
        if docs is None:
            return

        document_batches_generator = docs.batch(
            traversal_paths=parameters.get('traversal_paths', self.traversal_paths),
            batch_size=parameters.get('batch_size', self.batch_size),
            require_attr='text',
        )

        for document_batch in document_batches_generator:
            text_batch = document_batch.texts

            language = parameters.get('language', self.language)
            embeddings = self.model.embed_sentences(text_batch, lang=language)
            for document, embedding in zip(document_batch, embeddings):
                document.embedding = embedding
