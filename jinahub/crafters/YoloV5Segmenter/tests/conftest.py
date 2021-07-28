import glob
import os

import cv2
import pytest
from jina import DocumentArray, Document

cur_dir = os.path.dirname(os.path.abspath(__file__))

@pytest.fixture(scope='package')
def build_da():
    def _build_da():
        return DocumentArray([
            Document(blob=cv2.imread(path), tags={'filename': path.split('/')[-1]})
            for path in glob.glob(os.path.join(cur_dir, 'data/img/*.jpg'))
        ])

    return _build_da
