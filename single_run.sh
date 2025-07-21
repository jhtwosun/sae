export HF_HOME=/mnt/tmp/hf_cache
export CUDA_VISIBLE_DEVICES=2
python demo.py --save_dir /mnt/tmp/debug_log/run2 --model_name EleutherAI/pythia-160m-deduped --layers 8 --architectures hierarchical_recursive