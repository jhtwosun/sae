git config submodule.dictionary_learning.url https://github.com/saprmarks/dictionary_learning.git

git submodule update --init --recursive
pip install -e .

env HF_HOME='/mnt/tmp/hf_cache'