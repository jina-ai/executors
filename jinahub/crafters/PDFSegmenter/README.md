# ✨ PDFSegmenter

PDFPlumberSegmenter is a segmenter used for extracting images and text as chunks from PDF data. It stores each images and text of each page as chunks separately.


## 🌱 Prerequisites

Some conditions to fulfill before running the executor

Install requirements:

`pip install -r requirements.txt`

## 🚀 Usages

### 🚚 Via JinaHub

#### using docker images
Use the prebuilt images from JinaHub in your python codes, 

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

#### using source codes
Use the source codes from JinaHub in your python codes,

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


### 📦️ Via Pypi

1. Install the `executors` package.

	```bash
	pip install git+https://github.com/jina-ai/executors
	```

1. Use `PDFSegmenter` in your code

   ```python
   from jina import Flow
   from jinahub.crafters.PDFSegmenter.pdf_segmenter import PDFSegmenter
   
   f = Flow().add(uses=PDFSegmenter)
   ```


### 🐳 Via Docker

1. Clone the repo and build the docker image

	```shell
	git clone https://github.com/jina-ai/executors
	cd executors/jinahub/crafters/PDFSegmenter
	docker build -t executor-pdf-crafter .
	```

1. Use `executor-pdf-crafter` in your codes

	```python
	from jina import Flow
	
	f = Flow().add(uses='docker://executor-pdf-crafter:latest')
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
