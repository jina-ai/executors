from typing import Optional, Tuple

import torch
from jina import DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger
from jina_commons.batching import get_docs_batch_generator
from transformers import CLIPFeatureExtractor, CLIPModel


class CLIPImageEncoder(Executor):
    """
    Encode image into embeddings.

    :param pretrained_model_name_or_path: Can be either:
        - A string, the model id of a pretrained CLIP model hosted
            inside a model repo on huggingface.co, e.g., 'openai/clip-vit-base-patch32'
        - A path to a directory containing model weights saved, e.g.,
            ./my_model_directory/
    :param base_feature_extractor: Base feature extractor for images.
        Defaults to ``pretrained_model_name_or_path`` if None
    :param use_default_preprocessing: Whether to use the `base_feature_extractor` on
        images (blobs) before encoding them. If you disable this, you must ensure
        that the images you pass in have the correct format, see the ``encode`` method
        for details.
    :param device: device that the model is on (should be "cpu", "cuda" or "cuda:X",
        where X is the index of the GPU on the machine)
    :param default_batch_size: fallback batch size in case there is no batch size
        sent in the request
    :param default_traversal_paths: fallback traversal path in case there is no
        traversal path sent in the request
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = "openai/clip-vit-base-patch32",
        base_feature_extractor: Optional[str] = None,
        use_default_preprocessing: bool = True,
        device: str = "cpu",
        default_batch_size: int = 32,
        default_traversal_paths: Tuple = ("r",),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.default_batch_size = default_batch_size
        self.default_traversal_paths = default_traversal_paths
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.use_default_preprocessing = use_default_preprocessing
        self.base_feature_extractor = (
            base_feature_extractor or pretrained_model_name_or_path
        )

        self.logger = JinaLogger(self.__class__.__name__)

        if device.startswith("cuda") and not torch.cuda.is_available():
            self.logger.warning(
                "You tried to use GPU but torch did not detect your"
                "GPU correctly. Defaulting to CPU. Check your CUDA installation!"
            )
            device = "cpu"

        self.device = device
        self.preprocessor = CLIPFeatureExtractor.from_pretrained(
            self.base_feature_extractor
        )
        self.model = CLIPModel.from_pretrained(self.pretrained_model_name_or_path)
        self.model.to(self.device).eval()

    @requests
    def encode(self, docs: Optional[DocumentArray], parameters: dict, **kwargs):
        """
        Encode all docs with images and store the encodings in the embedding
        attribute of the docs.

        :param docs: documents sent to the encoder. The docs must have `blob` of the
            shape ``Height x Width x 3``. By default, the input ``blob`` must
            be an ``ndarray`` with ``dtype=uint8`` or ``dtype=float32``.
            The ``Height`` and ``Width`` can have arbitrary values.

            If you set ``use_default_preprocessing=True`` when creating this encoder,
            then the image arrays should have the shape ``[H, W, C]``, and be in the
            RGB color format.

            If you set ``use_default_preprocessing=False`` when creating this encoder,
            then you need to ensure that the images you pass in are already
            pre-processed. This means that they are all the same size (for batching) -
            the CLIP model was trained on ``224 x 224`` images, and that they are of
            the shape ``[C, H, W]`` (in the RGB color format). They should also be
            normalized.
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
                    blob_batch = [d.blob for d in batch_docs]
                    if self.use_default_preprocessing:
                        tensor = self._generate_input_features(blob_batch.copy())
                    else:
                        tensor = {
                            "pixel_values": torch.tensor(
                                blob_batch.copy(),
                                dtype=torch.float32,
                                device=self.device,
                            )
                        }

                    embeddings = self.model.get_image_features(**tensor)
                    embeddings = embeddings.cpu().numpy()

                    for doc, embed in zip(batch_docs, embeddings):
                        doc.embedding = embed

    def _generate_input_features(self, images):
        input_tokens = self.preprocessor(
            images=images,
            return_tensors="pt",
        )
        input_tokens = {
            k: v.to(torch.device(self.device)) for k, v in input_tokens.items()
        }
        return input_tokens
