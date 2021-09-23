# DPRTextEncoder

 **DPRTextEncoder** is a class that encodes text into embeddings using the DPR model from huggingface transformers.

The **DPR** model was originally proposed in [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906).

This encoder supports both the DPR context and question encoders - you should specify which type you are using with the `encoder_type` parameter.


## Reference


- [DPR paper](https://arxiv.org/abs/2004.04906)
- [Huggingface transformers DPR model documentation](https://huggingface.co/transformers/model_doc/dpr.html)
