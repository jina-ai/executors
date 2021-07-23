#!/bin/sh
MODEL_DIR=models
DATA_DIR=data
REPO_DIR=audioset_tagging_cnn

mkdir -p ${MODEL_DIR}

if [ ! -f "${MODEL_DIR}/vggish_model.ckpt" ]; then
  echo "Downloading model"
  echo "------ Download Vggish model ------"
  curl https://storage.googleapis.com/audioset/vggish_model.ckpt --output ${MODEL_DIR}/vggish_model.ckpt
  echo "------ Download PCA model ------"
  curl https://storage.googleapis.com/audioset/vggish_pca_params.npz --output ${MODEL_DIR}/vggish_pca_params.npz
else
  echo "Model already exists! Skipping."
fi