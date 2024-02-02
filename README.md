# SimpleImageGeneration

Simple implementation of awesome image generation methods:

- vae
- gan
- flow
- diffusion
- auto_regressive

## Qucik Start

只有一些通用的 utils、datasets 等放在了 simgen 包目录里面方便复用，其他的与各生成方法强相关的 models、loss 均在子目录中各自实现

首先拉取本仓库并安装 simgen:

```shell
git clone git@github.com:Adenialzz/SimpleImageGeneration.git
cd SimpleImageGeneration
pip install -e .
```

然后到各个子目录中开始训练，具体可再参照各个子目录中的 README
```shell
cd auto_regressive/vqvae
python main.py configs/mnist.yml
```
