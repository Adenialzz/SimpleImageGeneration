
class RescaleChannels(object):  # 将像素值从范围 [0.0, 1.0] 转换为 [-1.0, 1.0]。这样做是为了使输入图像的值范围大致与标准高斯分布的范围相同。
    def __call__(self, sample):
        return 2 * sample - 1
    
class UnrescaleChannels(object):
    def __call__(self, sample):
        return ((sample + 1) / 2).clip(0, 1)
