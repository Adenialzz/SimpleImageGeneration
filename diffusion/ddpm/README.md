
**WARNING: This repo may has some bugs, the generation result is not good as [abarankab/ddpm](https://github.com/abarankab/DDPM). This BUG will be fixed in the near future.**

# DDPM

DDPM在CIFAR10上进行训练/采样的代码，带有diffusion实现的公式注释，参考：[DDPM详解](https://zhuanlan.zhihu.com/p/673353348)。


#### 启动训练
```shell
python train.py configs/ddpm_cifar10.yaml
```

#### 采样生成
```shell
python sample.py configs/ddpm_cifar10.yaml
```

#### TODO

1. 看看类别引导的实现：
```python
self.class_bias = nn.Embedding(num_classes, out_channels)
```


参考实现：[abarankab/ddpm](https://github.com/abarankab/DDPM)

扩展实现：[Stability-AI/ddpm](https://github.com/Stability-AI/stablediffusion/blob/main/ldm/models/diffusion/ddpm.py)

