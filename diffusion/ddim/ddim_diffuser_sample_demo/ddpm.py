from diffusers import DDIMPipeline

# DDPM训练的模型，可直接同DDIM进行采样加速
model_id = "google/ddpm-cifar10-32"  

# 加载model和pipeline
ddim = DDIMPipeline.from_pretrained(model_id)

# DDIM采样生成，设置总步数为50
image = ddim(num_inference_steps=50).images[0]

# 保存图片
image.save("ddim_generated_image.png")