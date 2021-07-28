#!/bin/sh
CACHE_DIR=.cache
BASE_URL=https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1
mkdir -p ${CACHE_DIR}

MODEL_FILE_NAME=AudioCLIP-Partial-Training.pt
if [ ! -f "${CACHE_DIR}/${MODEL_FILE_NAME}" ]; then
  echo "------ Downloading AudioCLIP model ------"
  wget ${BASE_URL}/${MODEL_FILE_NAME} -O ${CACHE_DIR}/${MODEL_FILE_NAME}
else
  echo "Model already exists! Skipping."
fi

VOCAB_FILE_NAME=bpe_simple_vocab_16e6.txt.gz
if [ ! -f "${CACHE_DIR}/${VOCAB_FILE_NAME}" ]; then
  echo "------ Downloading vocab ------"
  wget ${BASE_URL}/${VOCAB_FILE_NAME} -O ${CACHE_DIR}/${VOCAB_FILE_NAME}
else
  echo "Vocab already exists! Skipping."
fi
