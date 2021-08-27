
# YoloV5Segmenter

**YoloV5Segmenter** is a class that wraps the [YoloV5](https://github.com/ultralytics/yolov5) model for generating bounding boxes from images and creating chunks. 








## Usage 

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


## Reference
- [Yolo paper](https://arxiv.org/abs/1506.02640v5)
- [YoloV5 code](https://github.com/ultralytics/yolov5)
