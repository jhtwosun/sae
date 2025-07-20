cd ..

sudo apt update && sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.10 python3.10-dev -y

python3 -m pip install --user -U virtualenv

PATH=$PATH:~/.local/bin
python3 -m virtualenv .dlr --python=python3.10
PATH=$PATH:~/.dlr/bin
source .dlr/bin/activate

cd sae


pip install -e .

env HF_HOME='/mnt/tmp/hf_cache'
echo HF_HOME='/mnt/tmp/hf_cache' >> ~/.bashrc

sudo apt install tmux -y

git config --global user.email "jhtwosun@gmail.com"
git config --global user.name "jhtwosun"