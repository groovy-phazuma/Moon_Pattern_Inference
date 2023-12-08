# settings
lst_lr=(
    3e-6
    1e-5
    3e-5
    1e-4
    3e-4
    1e-3
)
# exec
cd ~/morita/coder
for lr in {0..5};do
    docker compose exec ctn python /workspace/Moon_Pattern_Inference/silhouette_model/train.py \
    --seed 24771 --dir_result ViT/lr${lst_lr[lr]}_b128 \
    --model_name ViT \
    --num_epoch 100 --batch_size 128 \
    --lr ${lst_lr[lr]} \
    --lr_min "1e-5" --warmup_t 10 --warmup_lr_init "1e-5" \
    --patience 25
done