# DPRTextEncoder

 **DPRTextEncoder** is a class that encodes text into embeddings using the DPR model from huggingface transformers.

The **DPR** model was originally proposed in [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906).

This encoder supports both the DPR context and question encoders - you should specify which type you are using with the `encoder_type` parameter.

If the `encoder_type` is `context`, please also specify the context encoder model by setting the `pretrained_model_name_or_path`.

As an example, if `encoder_type` is `context`, you may set `pretrained_model_name_or_path` to `facebook/dpr-ctx_encoder-single-nq-base`.



## Reference

- [DPR paper](https://arxiv.org/abs/2004.04906)
- [Huggingface transformers DPR model documentation](https://huggingface.co/transformers/model_doc/dpr.html)

<!-- version=v0.4 -->
