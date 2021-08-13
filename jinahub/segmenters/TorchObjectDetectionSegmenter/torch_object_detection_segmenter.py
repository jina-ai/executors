__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Dict, List, Tuple, Optional

import numpy as np
import torchvision.models.detection as detection_models

from jina import Document, DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger
from jina_commons.batching import get_docs_batch_generator
from jina_commons.encoders.image.preprocessing import load_image, crop_image, move_channel_axis


class TorchObjectDetectionSegmenter(Executor):
    """
    :class:`TorchObjectDetectionSegmenter` detects objects
    from an image using `torchvision detection models`
    and crops the images according tothe detected bounding boxes
    of the objects with a confidence higher than a threshold.
    :param on_gpu: set to True if using GPU
    :param model_name: the name of the model. Supported models include
        ``fasterrcnn_resnet50_fpn``, ``maskrcnn_resnet50_fpn`
    :param confidence_threshold: confidence value from which it
        considers a positive detection and therefore the object detected will be cropped and returned
    :param label_name_map: A Dict mapping from label index to label name, by default will be
        COCO_INSTANCE_CATEGORY_NAMES
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
        TODO: Allow changing the backbone
    """
    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    def __init__(self, model_name: Optional[str] = None,
                 on_gpu: bool = False,
                 default_traversal_paths: Tuple = ('r', ),
                 default_batch_size: int = 32,
                 confidence_threshold: float = 0.0,
                 label_name_map: Optional[Dict[int, str]] = None,
                 *args, **kwargs):
        """Set constructor"""
        super().__init__(*args, **kwargs)
        self.logger = JinaLogger(self.__class__.__name__)
        self.on_gpu = on_gpu
        self.model_name = model_name or 'fasterrcnn_resnet50_fpn'
        self._default_channel_axis = 0
        self.default_batch_size = default_batch_size
        self.default_traversal_paths = default_traversal_paths
        self.confidence_threshold = confidence_threshold
        self.label_name_map = label_name_map or TorchObjectDetectionSegmenter.COCO_INSTANCE_CATEGORY_NAMES
        self.model = getattr(detection_models, self.model_name)(pretrained=True, pretrained_backbone=True).eval()

    def _predict(self, batch: List[np.ndarray]) -> 'np.ndarray':
        """
        Run the model for prediction
        :param img: the image from which to run a prediction
        :return: the boxes, scores and labels predictedx
        """
        import torch
        _input = torch.from_numpy(np.stack(batch).astype('float32'))

        if self.on_gpu:
            _input = _input.cuda()

        return self.model(_input)

    @requests
    def segment(self, docs: DocumentArray, parameters: dict, *args, **kwargs):
        """
        Crop the input image array within DocumentArray.
        :param docs: docs containing the ndarrays of the images
        :return: a list of chunk dicts with the cropped images
        :param args:  Additional positional arguments e.g. traversal_paths, batch_size
        :param kwargs: Additional keyword arguments
        """
        if docs:
            # traverse through a generator of batches of docs
            for docs_batch in get_docs_batch_generator(
                docs,
                traversal_path=parameters.get('traversal_paths', self.default_traversal_paths),
                batch_size=parameters.get('batch_size', self.default_batch_size),
                needs_attr='blob'
            ):
                # the blob dimension of imgs/cars.jpg at this point is (2, 681, 1264, 3)
                # Ensure the color channel axis is the default axis. i.e. c comes first
                # e.g. (h,w,c) -> (c,h,w) / (b,h,w,c) -> (b,c,h,w)
                blob_batch = [move_channel_axis(d.blob, -1,
                              self._default_channel_axis) for d in docs_batch]
                all_predictions = self._predict(blob_batch)

                for doc, blob, predictions in zip(docs_batch, blob_batch, all_predictions):
                    bboxes = predictions['boxes'].detach()
                    scores = predictions['scores'].detach()
                    labels = predictions['labels']
                    if self.on_gpu:
                        bboxes = bboxes.cpu()
                        scores = scores.cpu()
                        labels = labels.cpu()
                    img = load_image(blob * 255, self._default_channel_axis)

                    for bbox, score, label in zip(bboxes.numpy(), scores.numpy(), labels.numpy()):
                        if score >= self.confidence_threshold:
                            x0, y0, x1, y1 = bbox
                            # note that tensors are [H, W] while PIL Images are [W, H]
                            top, left = int(y0), int(x0)
                            # target size must be (h, w)
                            target_size = (int(y1) - int(y0), int(x1) - int(x0))
                            # at this point, raw_img has the channel axis at the default tensor one
                            _img, top, left = crop_image(img, target_size=target_size, top=top, left=left, how='precise')
                            _img = np.asarray(_img).astype('float32')
                            label_name = self.label_name_map[label]
                            self.logger.debug(
                                f'detected {label_name} with confidence {score} at position {(top, left)} and size {target_size}')

                            # a chunk is created for each of the objects detected for each image
                            d = Document(offset=0, weight=1., blob = _img, location=[top, left], tags={'label': label_name})
                            doc.chunks.append(d)
