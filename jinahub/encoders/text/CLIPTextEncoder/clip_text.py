from typing import Dict, Optional, Sequence

import torch
from jina import DocumentArray, Executor, requests
from jina_commons.batching import get_docs_batch_generator
from transformers import CLIPModel, CLIPTokenizer


class CLIPTextEncoder(Executor):
    """Encode text into embeddings using the CLIP model."""

    def __init__(
        self,
        pretrained_model_name_or_path: str = 'openai/clip-vit-base-patch32',
        base_tokenizer_model: Optional[str] = None,
        max_length: int = 77,
        device: str = 'cpu',
        traversal_paths: Sequence[str] = ['r'],
        batch_size: int = 32,
        *args,
        **kwargs,
    ):
        """
        :param pretrained_model_name_or_path: Can be either:
            - A string, the model id of a pretrained CLIP model hosted
                inside a model repo on huggingface.co, e.g., 'openai/clip-vit-base-patch32'
            - A path to a directory containing model weights saved, e.g., ./my_model_directory/
        :param base_tokenizer_model: Base tokenizer model.
            Defaults to ``pretrained_model_name_or_path`` if None
        :param max_length: Max length argument for the tokenizer.
            All CLIP models use 77 as the max length
        :param device: Pytorch device to put the model on, e.g. 'cpu', 'cuda', 'cuda:1'
        :param traversal_paths: Default traversal paths for encoding, used if
            the traversal path is not passed as a parameter with the request.
        :param batch_size: Default batch size for encoding, used if the
            batch size is not passed as a parameter with the request.
        """
        super().__init__(*args, **kwargs)
        self.default_traversal_paths = traversal_paths
        self.default_batch_size = batch_size
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.base_tokenizer_model = (
            base_tokenizer_model or pretrained_model_name_or_path
        )
        self.max_length = max_length

        self.device = device
        self.tokenizer = CLIPTokenizer.from_pretrained(self.base_tokenizer_model)
        self.model = CLIPModel.from_pretrained(self.pretrained_model_name_or_path)
        self.model.eval().to(device)

    @requests
    def encode(self, docs: Optional[DocumentArray], parameters: Dict, **kwargs):
        """
        Encode all documents with the `text` attribute and store the embeddings in the
        `embedding` attribute.

        :param docs: DocumentArray containing the Documents to be encoded
        :param parameters: A dictionary that contains parameters to control encoding.
            The accepted keys are ``traversal_paths`` and ``batch_size`` - in their
            absence their corresponding default values are used.
        """
        for docs_batch in get_docs_batch_generator(
            docs,
            traversal_path=parameters.get(
                'traversal_paths', self.default_traversal_paths
            ),
            batch_size=parameters.get('batch_size', self.default_batch_size),
            needs_attr='text',
        ):
            text_batch = docs_batch.get_attributes('text')

            with torch.no_grad():
                input_tokens = self._generate_input_tokens(text_batch)
                embeddings = self.model.get_text_features(**input_tokens).cpu().numpy()
                for doc, embedding in zip(docs_batch, embeddings):
                    doc.embedding = embedding

    def _generate_input_tokens(self, texts: Sequence[str]):

        input_tokens = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding='longest',
            truncation=True,
            return_tensors='pt',
        )
        input_tokens = {k: v.to(self.device) for k, v in input_tokens.items()}
        return input_tokens
