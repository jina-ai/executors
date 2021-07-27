__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import io
from typing import List

import fitz
import numpy as np
import pdfplumber
from jina import DocumentArray, Executor, requests, Document
from jina.logging.logger import JinaLogger


class PDFCrafter(Executor):
    """
    :class:`PDFCrafter` Extracts data (text and images) from PDF files.
    Stores images (`mime_type`=image/*) on chunk level ('c') and text segments (`mime_type`=text/plain)
    on chunk level ('c') in the root ('r') Document.
    """

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.logger = JinaLogger(context=self.__class__.__name__)

    @requests
    def craft(self, docs: DocumentArray, **kwargs):
        """
        Read PDF files. Extracts data from them.
        Checks if the input is a string of the filename,
        or if it's the file in bytes.
        It will then extract the data from the file, creating a list for images,
        and text.
        :param docs: Array of Documents.
        """
        for doc in docs:
            pdf_img, pdf_text = self._parse_pdf(doc)

            if pdf_img is not None:
                images = self._extract_image(pdf_img)
                doc.chunks.extend([Document(blob=img, mime_type='image/*') for img in images])
            if pdf_text is not None:
                texts = self._extract_text(pdf_text)
                doc.chunks.extend([Document(text=t, mime_type='text/plain') for t in texts])

    def _parse_pdf(self, doc: Document):
        pdf_img = None
        pdf_text = None
        try:
            if doc.uri:
                pdf_img = fitz.open(doc.uri)
                pdf_text = pdfplumber.open(doc.uri)
            if doc.buffer:
                pdf_img = fitz.open(stream=doc.buffer, filetype='pdf')
                pdf_text = pdfplumber.open(io.BytesIO(doc.buffer))
        except Exception as ex:
            self.logger.error(f'Failed to open due to: {ex}')
        return pdf_img, pdf_text

    def _extract_text(self, pdf_text) -> List[str]:
        # Extract text
        with pdf_text:
            texts = []
            count = len(pdf_text.pages)
            for i in range(count):
                page = pdf_text.pages[i]
                texts.append(page.extract_text(x_tolerance=1, y_tolerance=1))
            return texts

    def _extract_image(self, pdf_img) -> List['np.ndarray']:
        with pdf_img:
            images = []
            for page in range(len(pdf_img)):
                for img in pdf_img.getPageImageList(page):
                    xref = img[0]
                    pix = fitz.Pixmap(pdf_img, xref)
                    # read data from buffer and reshape the array into 3-d format
                    np_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n).astype('float32')
                    if pix.n - pix.alpha < 4:  # if gray or RGB
                        if pix.n == 1:  # convert gray to rgb
                            images.append(np.concatenate((np_arr,) * 3, -1))
                        elif pix.n == 4:  # remove transparency layer
                            images.append(np_arr[..., :3])
                        else:
                            images.append(np_arr)
                    else:  # if CMYK:
                        pix = fitz.Pixmap(fitz.csRGB, pix)  # Convert to RGB
                        np_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n).astype(
                            'float32')
                        images.append(np_arr)
        return images
