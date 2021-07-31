from typing import Optional, Tuple

import torch
from jina import Executor, DocumentArray, requests
from jina.logging.logger import JinaLogger
from jina_commons.batching import get_docs_batch_generator
from transformers import CLIPProcessor, CLIPModel


class CLIPImageEncoder(Executor):
    """
    Encode image into embeddings.

    :param pretrained_model_name_or_path: Can be either:
        - A string, the model id of a pretrained CLIP model hosted
            inside a model repo on huggingface.co, e.g., 'openai/clip-vit-base-patch32'
        - A path to a directory containing model weights saved, e.g., ./my_model_directory/
    :param processor: a CLIP processor which wraps a CLIP feature extractor and a CLIP
        tokenizer into a single processor. Defaults to ``pretrained_model_name_or_path`` if None
    :param device: device that the model is on (should be "cpu", "cuda" or "cuda:X",
        where X is the index of the GPU on the machine)
    :param default_batch_size: fallback batch size in case there is no batch size sent in the request
    :param default_traversal_paths: fallback traversal path in case there is no traversal path sent in the request
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = "openai/clip-vit-base-patch32",
        processor: Optional[str] = None,
        device: Optional[str] = "cpu",
        default_batch_size: int = 32,
        default_traversal_paths: Tuple = ("r",),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.default_batch_size = default_batch_size
        self.default_traversal_paths = default_traversal_paths
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.processor = processor or pretrained_model_name_or_path

        self.logger = JinaLogger(self.__class__.__name__)

        if device.startswith("cuda") and not torch.cuda.is_available():
            self.logger.warning(
                "You tried to use GPU but torch did not detect your"
                "GPU correctly. Defaulting to CPU. Check your CUDA installation!"
            )
            device = "cpu"

        self.device = device
        self.pre_processor = CLIPProcessor.from_pretrained(self.processor)
        self.model = CLIPModel.from_pretrained(self.pretrained_model_name_or_path)
        self.model.to(torch.device(device)).eval()

    @requests
    def encode(self, docs: Optional[DocumentArray], parameters: dict, **kwargs):
        """
        Encode all docs with images and store the encodings in the embedding attribute of the docs.

        :param docs: documents sent to the encoder. The docs must have `blob` of the
            shape `Height x Width x 3`. By default, the input `blob` must be an `ndarray`
            with `dtype=uint8`. The `Height` and `Width` can have arbitrary values.
        :param parameters: A dictionary that contains parameters to control encoding.
            The accepted keys are ``traversal_paths`` and ``batch_size`` - in their
            absence their corresponding default values are used.
        """
        if docs:
            document_batches_generator = get_docs_batch_generator(
                docs,
                traversal_path=parameters.get(
                    "traversal_paths", self.default_traversal_paths
                ),
                batch_size=parameters.get("batch_size", self.default_batch_size),
                needs_attr="blob",
            )

            with torch.no_grad():
                for batch_docs in document_batches_generator:
                    images = []
                    for doc in batch_docs:
                        if doc.blob.shape[2] != 3:
                            raise ValueError(
                                "Your image must be of the format "
                                "[H, W, C], in the RGB format (C=3),"
                                f" but got C={doc.blob.shape[2]} instead."
                            )
                        images.append(doc.blob)

                    tensor = self.pre_processor(images=images, return_tensors="pt")
                    tensor = tensor.to(self.device)
                    embeddings = self.model.get_image_features(**tensor)
                    embeddings = embeddings.cpu().numpy()
                    for idx, doc in enumerate(batch_docs):
                        doc.embedding = embeddings[idx]
