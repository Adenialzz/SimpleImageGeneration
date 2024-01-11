from diffusers import StableDiffusionPipeline, DDIMScheduler

ddim_scheduler = DDIMScheduler.from_config("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", scheduler=ddim_scheduler)

image = pipeline("An astronaut riding a horse.").images[0]

image.save("astronaut_riding_a_horse.png")