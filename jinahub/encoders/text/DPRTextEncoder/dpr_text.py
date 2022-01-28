from typing import Optional

import torch
from jina import DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizerFast,
)


class DPRTextEncoder(Executor):
    """
    Encode text into embeddings using a DPR model. You have to choose
    whether to use a context or a question encoder.

    For context encoders it is recommened to encode them together with the title,
    by setting the ``title_tag_key`` property. This is in order to match the
    encoding method used in model training.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = 'facebook/dpr-question_encoder-single-nq-base',
        encoder_type: str = 'question',
        base_tokenizer_model: Optional[str] = None,
        title_tag_key: Optional[str] = None,
        max_length: Optional[int] = None,
        traversal_paths: str = '@r',
        batch_size: int = 32,
        device: str = 'cpu',
        *args,
        **kwargs,
    ):
        """
        :param pretrained_model_name_or_path: Can be either:
            - the model id of a pretrained model hosted inside a model repo
              on [huggingface.co](huggingface.co).
            - A path to a directory containing model weights, saved using
              the transformers model's ``save_pretrained()`` method
        :param encoder_type: Either ``'context'`` or ``'question'``. Make sure this
            matches the model that you are using.
        :param base_tokenizer_model: Base tokenizer model. The possible values are
            the same as for the ``pretrained_model_name_or_path`` parameters. If not
            provided, the ``pretrained_model_name_or_path`` parameter value will be used
        :param title_tag_key: The key under which the titles are saved in the documents'
            tag property. It is recommended to set this property for context encoders,
            to match the model pre-training. It has no effect for question encoders.
        :param max_length: Max length argument for the tokenizer
        :param traversal_paths: Default traversal paths for encoding, used if the
            traversal path is not passed as a parameter with the request.
        :param batch_size: Default batch size for encoding, used if the
            batch size is not passed as a parameter with the request.
        :param device: The device (cpu or gpu) that the model should be on.
        """
        super().__init__(*args, **kwargs)
        self.device = device
        self.max_length = max_length
        self.title_tag_key = title_tag_key
        self.logger = JinaLogger(self.__class__.__name__)

        if encoder_type not in ['context', 'question']:
            raise ValueError(
                'The ``encoder_type`` parameter should be either "context"'
                f' or "question", but got {encoder_type}'
            )
        self.encoder_type = encoder_type

        if (
            encoder_type == 'context'
            and pretrained_model_name_or_path
            == 'facebook/dpr-question_encoder-single-nq-base'
        ):
            raise ValueError(
                'Encoder type is context but pretrained model is not set and '
                f'default model {pretrained_model_name_or_path} is a question model. '
                'Please ensure that pretrained_model_name_or_path is correctly set '
                'to a dpr context encoder model. e.g. "facebook/dpr-ctx_encoder-single-nq-base" '
            )

        if not base_tokenizer_model:
            base_tokenizer_model = pretrained_model_name_or_path

        if encoder_type == 'context':
            if not self.title_tag_key:
                self.logger.warning(
                    'The `title_tag_key` argument is not set - it is recommended'
                    ' to encode the context text together with the title to match the'
                    ' model pre-training. '
                )
            self.tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(
                base_tokenizer_model
            )
            self.model = DPRContextEncoder.from_pretrained(
                pretrained_model_name_or_path
            )
        elif encoder_type == 'question':
            self.tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(
                base_tokenizer_model
            )
            self.model = DPRQuestionEncoder.from_pretrained(
                pretrained_model_name_or_path
            )

        self.model = self.model.to(self.device).eval()

        self.traversal_paths = traversal_paths
        self.batch_size = batch_size

    @requests
    def encode(
        self, docs: Optional[DocumentArray] = None, parameters: dict = {}, **kwargs
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

        if docs is None:
            return

        document_batches_generator = DocumentArray(
            filter(
                lambda x: bool(x.text),
                docs[parameters.get('traversal_paths', self.traversal_paths)],
            )
        ).batch(batch_size=parameters.get('batch_size', self.batch_size))

        for batch_docs in document_batches_generator:
            with torch.inference_mode():
                texts = batch_docs.texts
                text_pairs = None
                if self.encoder_type == 'context' and self.title_tag_key:
                    text_pairs = list(
                        filter(
                            lambda x: x is not None,
                            batch_docs[:, f'tags__{self.title_tag_key}'],
                        )
                    )
                    if len(text_pairs) != len(batch_docs):
                        raise ValueError(
                            'If you set `title_tag_key` property, all documents'
                            ' that you want to encode must have this tag. Found'
                            f' {len(text_pairs) - len(batch_docs)} documents'
                            ' without it.'
                        )

                inputs = self.tokenizer(
                    text=texts,
                    text_pair=text_pairs,
                    max_length=self.max_length,
                    padding='longest',
                    truncation=True,
                    return_tensors='pt',
                ).to(self.device)
                embeddings = self.model(**inputs).pooler_output.cpu().numpy()

            batch_docs.embeddings = embeddings
