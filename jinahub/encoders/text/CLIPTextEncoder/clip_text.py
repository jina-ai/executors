import os
from typing import Dict, List, Optional

import torch
from jina import DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger
from jina_commons.batching import get_docs_batch_generator
from transformers import CLIPTokenizer, CLIPModel


class CLIPTextEncoder(Executor):
    """Encode text into embeddings using a CLIP model.

    :param pretrained_model_name_or_path: Can be either:
        - A string, the model id of a pretrained CLIP model hosted
            inside a model repo on huggingface.co, e.g., 'openai/clip-vit-base-patch32'
        - A path to a directory containing model weights saved, e.g., ./my_model_directory/
    :param base_tokenizer_model: Base tokenizer model.
        Defaults to ``pretrained_model_name_or_path`` if None
    :param max_length: Max length argument for the tokenizer.
        All CLIP models use 77 as the max length
    :param device: Device to be used. Use 'cuda' for GPU.
    :param default_traversal_paths: Default traversal paths for encoding, used if the
        traversal path is not passed as a parameter with the request.
    :param default_batch_size: Default batch size for encoding, used if the
        batch size is not passed as a parameter with the request.
    :param args: Arguments
    :param kwargs: Keyword Arguments
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = 'openai/clip-vit-base-patch32',
        base_tokenizer_model: Optional[str] = None,
        max_length: Optional[int] = 77,
        device: str = 'cpu',
        default_traversal_paths: List[str] = ['r'],
        default_batch_size: int = 32,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.default_traversal_paths = default_traversal_paths
        self.default_batch_size = default_batch_size
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.base_tokenizer_model = (
            base_tokenizer_model or pretrained_model_name_or_path
        )
        self.max_length = max_length
        self.logger = JinaLogger(self.__class__.__name__)

        if device.startswith('cuda') and not torch.cuda.is_available():
            self.logger.warning(
                'You tried to use GPU but torch did not detect your'
                'GPU correctly. Defaulting to CPU. Check your CUDA installation!'
            )
            device = 'cpu'

        self.device = device
        self.tokenizer = CLIPTokenizer.from_pretrained(self.base_tokenizer_model)
        self.model = CLIPModel.from_pretrained(self.pretrained_model_name_or_path)
        self.model.eval().to(torch.device(device))

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
                embedding_batch = self.model.get_text_features(**input_tokens)
                numpy_embedding_batch = embedding_batch.cpu().numpy()
                for document, numpy_embedding in zip(
                    document_batch, numpy_embedding_batch
                ):
                    document.embedding = numpy_embedding

    def _generate_input_tokens(self, texts):

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
