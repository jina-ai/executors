# VADSpeechSegmenter

**VADSpeechSegmenter** segments speech audio using Silero VAD (voice activity detector).


Instead of segmenting according to time at fixed-length intervals, this segmenter segments based on voice activity.


This VAD is similar to WebRTC but instead of differentiating voice from silence, this segmenter differenciates voice from noise/music/silence.






## Usage

```python
import os
import shutil
from pathlib import Path

import requests
from jina import Flow, Document
from executors import VADSpeechSegmenter

cur_dir = os.path.dirname(os.path.abspath(__file__))
workspace = Path(cur_dir) / 'workspace'

path = 'test.wav'
req = requests.get('https://www.ee.columbia.edu/~dpwe/sounds/musp/msms1.wav', stream=True)
with open(path, 'wb') as f:
    shutil.copyfileobj(req.raw, f)

f = Flow().add(uses=VADSpeechSegmenter,
               uses_with={'normalize': False, 'dump': True},
               uses_meta={'workspace': str(workspace / 'segmenter')})

with f:
    resp = f.post(on='/segment', inputs=Document(uri=path), return_results=True)
    print(resp)
```
The segmented audio along with the original audio files can be found in the `{workspace}/audio` path.

### Inputs

`Document` whose `blob` stores the audio in numpy array OR whose `uri` stores the path to the file of the audio that ends with either `mp3` or `wav`.

### Returns

`Document` with `chunks` that contain the segmented audio with `location` that denotes the start and end index of the chunk in the original data.


## Reference
- [silero-vad](https://github.com/snakers4/silero-vad)
