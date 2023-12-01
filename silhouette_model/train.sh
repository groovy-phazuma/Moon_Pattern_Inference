# 
cd ~/morita/coder
docker compose exec ctn python /workspace/Moon_Pattern_Inference/silhouette_model/train.py \
--seed 24771 --dir_result ViT_test \
--model_name ViT \
--num_epoch 100 --batch_size 512 \
--lr "1e-3" \
--lr_min "1e-5" --warmup_t 5 --warmup_lr_init "1e-5"