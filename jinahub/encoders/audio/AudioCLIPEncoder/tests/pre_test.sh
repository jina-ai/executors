sudo apt-get update && sudo apt-get install -y libsndfile1-dev
cd $(dirname "$BASH_SOURCE")/..
bash scripts/download_model.sh