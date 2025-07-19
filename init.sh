python3 -m virtualenv .dlr

source .dlr/bin/activate

cd dictionary_learning_demo

git config submodule.dictionary_learning.url https://github.com/saprmarks/dictionary_learning.git

git submodule update --init --recursive
pip install -e .

env HF_HOME='/mnt/tmp/hf_cache'
echo HF_HOME='/mnt/tmp/hf_cache' >> ~/.bashrc
source ~/.bashrc