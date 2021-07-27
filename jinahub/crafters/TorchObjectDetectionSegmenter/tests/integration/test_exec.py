__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from jina import Flow, Document
from jinahub.segmenter.torch_object_detection_segmenter import TorchObjectDetectionSegmenter


def test_exec():
    f = Flow().add(uses=TorchObjectDetectionSegmenter)
    with f:
        resp = f.post(on='/test', inputs=Document(), return_results=True)
        assert resp is not None
