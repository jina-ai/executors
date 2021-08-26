# ✨ PDFSegmenter 

PDFPlumberSegmenter is a segmenter used for extracting images and text as chunks from PDF data. It stores each images and text of each page as chunks separately.

- [✨ PDFSegmenter](#-pdfsegmenter)
  - [🌱 Prerequisites](#-prerequisites)
  - [🚀 Usages](#-usages)
    - [🚚 Via JinaHub](#-via-jinahub)
      - [using docker images](#using-docker-images)
      - [using source code](#using-source-code)
  - [🎉️ Example](#️-example)
    - [Inputs](#inputs)
    - [Returns](#returns)
  - [🔍️ Reference](#️-reference)

## 🌱 Prerequisites

> These are only needed if you download the source code and directly use the class. Not needed if you use the Jina Hub method below.

In case you want to install the dependencies locally run 
```
pip install -r requirements.txt
```

## 🚀 Usages

### 🚚 Via JinaHub

#### using docker images
Use the prebuilt images from JinaHub in your Python code: 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://PDFSegmenter')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: crafter
    uses: 'jinahub+docker://PDFSegmenter'
```

#### using source code
Use the source code from JinaHub in your Python code:

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://PDFSegmenter')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: crafter
    uses: 'jinahub://PDFSegmenter'
```

## 🎉️ Example 


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

## 🔍️ Reference
