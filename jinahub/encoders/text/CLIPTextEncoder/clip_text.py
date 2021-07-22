from jina import DocumentArray, Executor, requests
import torch
import clip
from typing import Iterable, Optional, List
from jina_commons.batching import get_docs_batch_generator


class CLIPTextEncoder(Executor):
    """Encode text into embeddings using a CLIP model.

    :param model_name: The name of one of the pre-trained CLIP models.
        Can also be a path to a local checkpoint (a ``.pt`` file).
    :param default_batch_size: Default batch size for encoding, used if the
        batch size is not passed as a parameter with the request.
    :param default_traversal_paths: Default traversal paths for encoding, used if the
        traversal path is not passed as a parameter with the request.
    :param default_device: The device (cpu or gpu) that the model should be on.
    :param jit: Whether a JIT version of the model should be loaded.
    """

    def __init__(
        self,
        model_name: str = 'ViT-B/32',
        default_batch_size: int = 32,
        default_traversal_paths: List[str] = ['r'],
        default_device: str = 'cpu',
        jit: bool = True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.device = default_device
        self.model, _ = clip.load(model_name, self.device, jit)
        self.default_traversal_paths = default_traversal_paths
        self.default_batch_size = default_batch_size

    @requests
    def encode(self, docs: Optional[DocumentArray], parameters: dict, **kwargs):
        """
        Encode all docs with text and store the encodings in the embedding
        attribute of the docs.

        :param docs: documents sent to the encoder. The docs must have text.
        :param parameters: dictionary to define the ``traversal_path`` and the
            ``batch_size``. For example,
            ``parameters={'traversal_paths': ['r'], 'batch_size': 10}``
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
        with torch.no_grad():
            for document_batch in document_batches_generator:
                text_batch = [d.text for d in document_batch]
                tensor = clip.tokenize(text_batch).to(self.device)
                embedding_batch = self.model.encode_text(tensor)
                numpy_embedding_batch = embedding_batch.cpu().numpy()
                for document, numpy_embedding in zip(
                    document_batch, numpy_embedding_batch
                ):
                    document.embedding = numpy_embedding
