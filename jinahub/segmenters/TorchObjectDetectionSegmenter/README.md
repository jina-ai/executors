# ✨ TorchObjectDetectionSegmenter

**TorchObjectDetectionSegmenter** is a class that supports object detection and bounding box extraction using PyTorch with Faster R-CNN and Mask R-CNN models.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [🌱 Prerequisites](#-prerequisites)
- [🚀 Usages](#-usages)
- [🎉️ Example](#%EF%B8%8F-example)
- [🔍️ Reference](#%EF%B8%8F-reference)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## 🌱 Prerequisites


> These are only needed if you download the source code and directly use the class. Not needed if you use the Jina Hub method below.

To install the dependencies locally run 
```
pip install -r requirements.txt
```

To verify the installation works:
```
pip install -r tests/requirements.txt
pytest -sv tests
```

## 🚀 Usages

### 🚚 Via JinaHub

#### using docker images
Use the prebuilt images from JinaHub in your Python code: 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://TorchObjectDetectionSegmenter')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://TorchObjectDetectionSegmenter'
```

#### using source code
Use the source code from JinaHub in your Python code:

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://TorchObjectDetectionSegmenter')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://TorchObjectDetectionSegmenter'
```


## 🎉️ Example 


```python
from jina import Flow, Document

f = Flow().add(uses='jinahub+docker://TorchObjectDetectionSegmenter')

with f:
    resp = f.post(on='foo', inputs=Document(), return_results=True)
	print(f'{resp}')
```

### Inputs 

`Document` whose `blob` stores the image to be detected with values between 0-1 and has color channel at the last axis.

### Returns

`Document` with `chunks` that contain the original image in `blob`, bounding box coordinates of objects detected in `location`, and image label key value pair in `tags`.


## 🔍️ Reference
- [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
- [Mask R-CNN](https://arxiv.org/abs/1703.06870)
- [TORCHVISION OBJECT DETECTION FINETUNING TUTORIAL](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)

