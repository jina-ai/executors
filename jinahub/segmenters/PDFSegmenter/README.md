# PDFSegmenter

PDFPlumberSegmenter is a segmenter used for extracting images and text as chunks from PDF data. It stores each images and text of each page as chunks separately.



## Usage 


```python
from jina import Flow, Document

f = Flow().add(uses='jinahub+docker://PDFSegmenter')

with f:
    resp = f.post(on='foo', inputs=Document(), return_results=True)
	print(f'{resp}')
```

### Inputs 

`Document` with `uri` or `buffer` of the PDF files. 

### Returns

`Document` with `chunks` containing text and images of the PDF

