__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os

from docarray import DocumentArray
from jina import Flow
from pdf_segmenter import PDFSegmenter
from PIL import Image


def test_flow(test_dir, doc_generator_img_text, expected_text):
    flow = Flow().add(uses=PDFSegmenter)
    doc_arrays = doc_generator_img_text
    for docs in doc_arrays:
        with flow:
            results = flow.post(on='/test', inputs=docs)

            assert len(results) == 1
            chunks = results[0].chunks
            assert len(chunks) == 3
            for idx, c in enumerate(chunks[:2]):
                with Image.open(
                    os.path.join(test_dir, f'data/test_img_{idx}.jpg')
                ) as img:
                    tensor = chunks[idx].tensor
                    assert chunks[idx].mime_type == 'image/*'
                    assert tensor.shape[1], tensor.shape[0] == img.size
                    if idx == 0:
                        assert tensor.shape == (660, 1024, 3)
                    if idx == 1:
                        assert tensor.shape == (626, 1191, 3)

                # Check text
                assert chunks[2].text == expected_text
                assert chunks[2].mime_type == 'text/plain'
