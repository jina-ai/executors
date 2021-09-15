import numpy as np
from jina import Flow, Document
from PIL import Image

f = Flow().add(uses='jinahub+docker://CLIPImageEncoder')

def print_result(resp):
    doc = resp.docs[0]
    print(f'Embedding image to {doc.embedding.shape[0]}-dimensional vector')

with f:
    doc = Document(blob=np.asarray(Image.open('myimage.png')))
    f.post(on='/foo', inputs=doc, on_done=print_result)

