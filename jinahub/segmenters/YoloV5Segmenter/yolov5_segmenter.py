__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import torch

from typing import Optional, Tuple, Dict, Iterable, List

from jina import Executor, DocumentArray, requests, Document
from jina_commons.batching import get_docs_batch_generator
from jina.logging.logger import JinaLogger

class YoloV5Segmenter(Executor):
    """
    Segment the image into bounding boxes and set labels

    :param model_name_or_path: the yolov5 model to use, can be a model name specified in ultralytics/yolov5's hubconf.py
    file, a custom model path or url
   :param default_batch_size: default batch size
   :param default_traversal_paths: default traversal path
   :param device: device to be used. Use 'cuda' for GPU
   :param size: image size used to perform inference
   :param augment: augment images during inference
   :param default_confidence_threshold: default confidence threshold

    """

    def __init__(self,
                 model_name_or_path: str = 'yolov5s',
                 default_batch_size: int = 32,
                 default_traversal_paths: Tuple = ('r',),
                 device: str = 'cpu',
                 size: int = 640,
                 augment: bool = False,
                 default_confidence_threshold: float = 0.3,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name_or_path = model_name_or_path
        self.default_batch_size = default_batch_size
        self.default_traversal_paths = default_traversal_paths
        self.logger = JinaLogger(self.__class__.__name__)
        self.default_size = size
        self.default_augment = augment
        self.default_confidence_threshold = default_confidence_threshold

        if device != 'cpu' and not device.startswith('cuda'):
            self.logger.error('Torch device not supported. Must be cpu or cuda!')
            raise RuntimeError('Torch device not supported. Must be cpu or cuda!')
        if device == 'cuda' and not torch.cuda.is_available():
            self.logger.warning(
                'You tried to use GPU but torch did not detect your'
                'GPU correctly. Defaulting to CPU. Check your CUDA installation!'
            )
            device = 'cpu'
        self.device = torch.device(device)
        self.model = self._load(self.model_name_or_path)

        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names


    @requests
    def segment(self, docs: Optional[DocumentArray], parameters: Dict, **kwargs):
        """
        Segment all docs into bounding boxes and set labels
        :param docs: documents sent to the segmenter. The docs must have `blob`.
        :param parameters: dictionary to override the default parameters. For example,
        `parameters={'traversal_paths': ['r'], 'batch_size': 10}` will override the `self.default_traversal_paths` and
        `self.default_batch_size`.
        """

        if docs:
            document_batches_generator = get_docs_batch_generator(
                docs,
                traversal_path=parameters.get('traversal_paths', self.default_traversal_paths),
                batch_size=parameters.get('batch_size', self.default_batch_size),
                needs_attr='blob'
            )
            self._segment_docs(document_batches_generator, parameters=parameters)

    def _segment_docs(self, document_batches_generator: Iterable, parameters: Dict):
        with torch.no_grad():
            for document_batch in document_batches_generator:
                images = [d.blob for d in document_batch]
                predictions = self.model(
                    images,
                    size=parameters.get('size', self.default_size),
                    augment=parameters.get('augment', self.default_augment)
                ).pred

                for doc, prediction in zip(document_batch, predictions):
                    for det in prediction:
                        x1, y1, x2, y2, conf, cls = det
                        if conf < parameters.get('confidence_threshold', self.default_confidence_threshold):
                            continue
                        c = int(cls)
                        crop = doc.blob[int(y1):int(y2), int(x1):int(x2), :]
                        doc.chunks.append(Document(
                                blob=crop,
                                tags={'label': self.names[c], 'conf': float(conf)}
                            ))

    def _load(self, model_name_or_path):
        if model_name_or_path in torch.hub.list('ultralytics/yolov5'):
            return torch.hub.load('ultralytics/yolov5', model_name_or_path, device=self.device)
        else:
            return torch.hub.load('ultralytics/yolov5', 'custom', model_name_or_path, device=self.device)
