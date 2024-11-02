
`denoising.py` 文件中包含了对[官方仓库采样方法](https://github.com/ermongroup/ddim/blob/main/functions/denoising.py)的注释，可结合下面的*参考实现*理解。

**TODO: 使用本目录的采样方法对训练好的DDPM进行采样生成，观察效果**

参考推导：[DDIM详解](https://zhuanlan.zhihu.com/p/674142410)

参考实现：DDIM Official Repo: [ermongroup/ddim](https://github.com/ermongroup/ddim)


### diffusers demo

**DDIM Pipeline**

```python
from diffusers import DDIMPipeline

# DDPM训练的模型，可直接同DDIM进行采样加速
model_id = "google/ddpm-cifar10-32"  

# 加载model和pipeline
ddim = DDIMPipeline.from_pretrained(model_id)

# DDIM采样生成，设置总步数为50
image = ddim(num_inference_steps=50).images[0]

# 保存图片
image.save("ddim_generated_image.png")
```


**Stable Diffusion with DDIM Scheduler**

```python
from diffusers import StableDiffusionPipeline, DDIMScheduler

ddim_scheduler = DDIMScheduler.from_config("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", scheduler=ddim_scheduler)

image = pipeline("An astronaut riding a horse.").images[0]

image.save("astronaut_riding_a_horse.png")
```