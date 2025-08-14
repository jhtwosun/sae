export HF_HOME=/mnt/tmp/hf_cache
export CUDA_VISIBLE_DEVICES=1
python demo.py --use_wandb --save_dir /mnt/tmp/sae_log/run2 --model_name EleutherAI/pythia-160m-deduped --layers 8 --architectures standard_new matryoshka_batch_top_k p_anneal hierarchical_batch_top_k hierarchical_batch_single_top_k