# ✨ PDFCrafter

PDFPlumberSegmenter is a segmenter used for extracting images and text as chunks from PDF data. It stores each images and text of each page as chunks separately.


## 🌱 Prerequisites

Some conditions to fulfill before running the executor

## 🚀 Usages

### 🚚 Via JinaHub

#### using docker images
Use the prebuilt images from JinaHub in your python codes, 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://PDFCrafter')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: crafter
    uses: 'jinahub+docker://PDFCrafter'
```

#### using source codes
Use the source codes from JinaHub in your python codes,

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://PDFCrafter')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: crafter
    uses: 'jinahub://PDFCrafter'
```


### 📦️ Via Pypi

1. Install the `executor-pdf-crafter` package.

	```bash
	pip install git+https://github.com/jina-ai/executor-pdf-crafter
	```

1. Use `executor-pdf-crafter` in your code

	```python
	from jina import Flow
	from jinahub.crafter.pdf_crafter import PDFCrafter
	
	f = Flow().add(uses=PDFCrafter)
	```


### 🐳 Via Docker

1. Clone the repo and build the docker image

	```shell
	git clone https://github.com/jina-ai/executor-pdf-crafter
	cd executor-pdf-crafter
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

f = Flow().add(uses='jinahub+docker://PDFCrafter')

with f:
    resp = f.post(on='foo', inputs=Document(), return_results=True)
	print(f'{resp}')
```

### Inputs 

`Document` with `uri` or `buffer` of the PDF files. 

### Returns

`Document` with `chunks` containing text and images of the PDF
