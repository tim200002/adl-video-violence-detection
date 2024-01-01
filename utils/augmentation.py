import torch
import numpy as np
from pytorchvideo.transforms import AugMix

def get_augmenter(batch):
    augmenter = AugMix()
    aug_array = torch.zeros_like(batch)

    for i, vid in enumerate(batch):
        # video has shape 3,16,172,172, we need to convert it to 16, 3, 172,172,
        vid = torch.transpose(vid, 0, 1)
        aug_vid = augmenter(vid)
        # change shape back
        aug_vid = torch.transpose(aug_vid, 0, 1)
        aug_array[i] = aug_vid
    
    return aug_array
    
    

# class VideoAugmetation(object):
#       def __call__(self, video):
#             video = video.numpy()

#             sometimes = lambda aug: va.Sometimes(0.1, aug)
#             seq = va.OneOf([
#             #va.RandomCrop(size=(240, 180)), # randomly crop video with a size of (240 x 180)
#             #va.RandomRotate(degrees=10), # randomly rotates the video with a degree randomly choosen from [-10, 10]
#             sometimes(va.GaussianBlur(sigma=1.0)),
#             sometimes(va.ElasticTransformation(alpha=0.5, sigma=1.0, order=3)),
#             sometimes(va.PiecewiseAffineTransform(displacement=1.0, displacement_kernel=1.0, displacement_magnification=1.0)),
#             sometimes(va.Superpixel( p_replace=30, n_segments=30)),
#             sometimes(va.Pepper(ratio=10)),

          
