


run training ans sampling

1. train vqvae as an image compressor

2. visualize reconstruction results for vqvae

3. train pixelcnn as generative model with fixed vqvae

4. sample and visualize final results with pixelcnn and vqvae


```python
python main.py configs/mnist.py
```

refrence from: [DL-Demos](https://github.com/SingleZombie/DL-Demos)