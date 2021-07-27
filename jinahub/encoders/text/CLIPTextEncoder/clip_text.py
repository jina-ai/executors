import os
from typing import Dict, List, Optional

import torch
from jina import DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger
from jina_commons.batching import get_docs_batch_generator
from transformers import CLIPTokenizer, CLIPModel


class CLIPTextEncoder(Executor):
    """Encode text into embeddings using a CLIP model.

    :param pretrained_model_name_or_path: The name of one of the pre-trained CLIP models.
        Can also be a path to a local checkpoint (a ``.pt`` file).
    :param base_tokenizer_model: Base tokenizer model
    :param max_length: Max length argument for the tokenizer
    :param embedding_fn_name: Function to call on the model in order to get output
    :param device: Device to be used. Use 'cuda' for GPU
    :param num_threads: The number of threads used for intraop parallelism on CPU
    :param default_traversal_paths: Default traversal paths for encoding, used if the
        traversal path is not passed as a parameter with the request.
    :param default_batch_size: Defines the batch size for inference on the loaded PyTorch model.
    :param args: Arguments
    :param kwargs: Keyword Arguments
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = 'openai/clip-vit-base-patch32',
        base_tokenizer_model: Optional[str] = None,
        max_length: Optional[int] = None,
        embedding_fn_name: str = 'get_text_features',
        device: str = 'cpu',
        num_threads: Optional[int] = None,
        default_traversal_paths: Optional[List[str]] = None,
        default_batch_size: int = 32,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if default_traversal_paths is not None:
            self.default_traversal_paths = default_traversal_paths
        else:
            self.default_traversal_paths = ['r']
        self.default_batch_size = default_batch_size
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.base_tokenizer_model = (
            base_tokenizer_model or pretrained_model_name_or_path
        )
        self.max_length = max_length
        self.logger = JinaLogger(self.__class__.__name__)

        if device not in ['cpu', 'cuda']:
            self.logger.error('Torch device not supported. Must be cpu or cuda!')
            raise RuntimeError('Torch device not supported. Must be cpu or cuda!')

        if device == 'cuda' and not torch.cuda.is_available():
            self.logger.warning(
                'You tried to use GPU but torch did not detect your'
                'GPU correctly. Defaulting to CPU. Check your CUDA installation!'
            )
            device = 'cpu'

        if device == 'cpu' and num_threads:
            cpu_num = os.cpu_count()
            if num_threads > cpu_num:
                self.logger.warning(
                    f'You tried to use {num_threads} threads > {cpu_num} CPUs'
                )
            else:
                torch.set_num_threads(num_threads)

        self.device = device
        self.embedding_fn_name = embedding_fn_name
        self.tokenizer = CLIPTokenizer.from_pretrained(self.base_tokenizer_model)
        self.model = CLIPModel.from_pretrained(self.pretrained_model_name_or_path)
        self.model.to(torch.device(device))

    @requests
    def encode(self, docs: Optional[DocumentArray], parameters: Dict, **kwargs):
        """
        Encode text data into a ndarray of `D` as dimension, and fill
        the embedding attribute of the docs.

        :param docs: DocumentArray containing text
        :param parameters: dictionary to define the `traversal_paths` and the `batch_size`. For example,
               `parameters={'traversal_paths': ['r'], 'batch_size': 10}`.
        :param kwargs: Additional key value arguments.
        """
        for document_batch in get_docs_batch_generator(
            docs,
            traversal_path=parameters.get(
                'traversal_paths', self.default_traversal_paths
            ),
            batch_size=parameters.get('batch_size', self.default_batch_size),
            needs_attr='text',
        ):
            text_batch = document_batch.get_attributes('text')

            with torch.no_grad():
                input_tokens = self._generate_input_tokens(text_batch)
                embedding_batch = getattr(self.model, self.embedding_fn_name)(
                    **input_tokens
                )
                if isinstance(embedding_batch, torch.Tensor):
                    numpy_embedding_batch = embedding_batch.cpu().numpy()
                for document, numpy_embedding in zip(
                    document_batch, numpy_embedding_batch
                ):
                    document.embedding = numpy_embedding

    def _generate_input_tokens(self, texts):
        if not self.tokenizer.pad_token:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer.vocab))

        input_tokens = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding='longest',
            truncation=True,
            return_tensors='pt',
        )
        input_tokens = {
            k: v.to(torch.device(self.device)) for k, v in input_tokens.items()
        }
        return input_tokens
