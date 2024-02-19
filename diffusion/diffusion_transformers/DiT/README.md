
adapted from [facebookresearch/DiT](https://github.com/facebookresearch/DiT)

sampling

```shell
python sample.py \
	--model DiT-XL/2 \
	--image-size 256 \
	--ckpt ./DiT-XL-2-256x256.pt \
```

start training

```shell
DATA_PATH=""

torchrun --nnodes=1 --nproc_per_node=2 train.py \
    --model DiT-S/8 \
    --vae mse \
    --data-path $DATA_PATH
```
