
# ✨ YoloV5Segmenter

**YoloV5Segmenter** is a class that wraps the [YoloV5](https://github.com/ultralytics/yolov5) model for generating bounding boxes from images and creating chunks. 

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [🌱 Prerequisites](#-prerequisites)
- [🚀 Usages](#-usages)
- [🎉️ Example](#%EF%B8%8F-example)
- [🔍️ Reference](#%EF%B8%8F-reference)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## 🌱 Prerequisites

Install dependencies using `pip install -r requirements.txt`.

## 🚀 Usages

### 🚚 Via JinaHub

#### using docker images
Use the prebuilt images from JinaHub in your python codes, 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://YoloV5Segmenter')
```

or in the `.yml` config.
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://YoloV5Segmenter'
```

#### using source codes
Use the source codes from JinaHub in your python codes,

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://YoloV5Segmenter')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://YoloV5Segmenter'
```


### 📦️ Via Pypi

1. Install the `jinahub-YoloV5Segmenter` package.

	```bash
	pip install git+https://github.com/jina-ai/executor-yolov5.git
	```

1. Use `jinahub-yolov5-segmenter` in your code

	```python
	from jina import Flow
	from jinahub.segmenter.yolov5_segmenter import YoloV5Segmenter
	
	f = Flow().add(uses='jinahub+docker://YoloV5Segmenter')
	```


### 🐳 Via Docker

1. Clone the repo and build the docker image

	```shell
	git clone https://github.com/jina-ai/executor-yolov5.git
	cd executor-yolov5
	docker build -t executor-yolov5-segmenter .
	```

1. Use `executor-yolov5-segmenter` in your codes

	```python
	from jina import Flow
	
	f = Flow().add(uses='docker://executor-yolov5-segmenter:latest')
	```

## 🎉️ Example 

Example with real data


```python
import cv2
import glob
from jina import Flow, Document, DocumentArray

f = Flow().add(uses='jinahub+docker://YoloV5Segmenter', timeout_ready=3000)

# Load data
doc_array = DocumentArray([
    Document(blob=cv2.imread("https://github.com/ultralytics/yolov5/blob/master/data/images/bus.jpg"))
])
with f:
    resp = f.post(on='test', inputs=doc_array, return_results=True)
    
print(f'{resp}')
```





### Inputs 

`Document` with `blob` of 3 dimensions containing the image.

### Returns

`Document` with `chunks` created that represent the detected bounding boxes. Each chunk has a blob of 3 dimensions and tags attribute containing the label (key `label`) and the confidence value (key `conf`).


## 🔍️ Reference
- [Yolo paper](https://arxiv.org/abs/1506.02640v5)
- [YoloV5 code](https://github.com/ultralytics/yolov5)
