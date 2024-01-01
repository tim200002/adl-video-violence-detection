import torch
import random
import torchvision
import torchvision.transforms as transforms


def crop(vid, i, j, h, w):
    return vid[..., i:(i + h), j:(j + w)]


def center_crop(vid, output_size):
    h, w = vid.shape[-2:]
    th, tw = output_size

    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(vid, i, j, th, tw)


def hflip(vid):
    return vid.flip(dims=(-1,))


# NOTE: for those functions, which generally expect mini-batches, we keep them
# as non-minibatch so that they are applied as if they were 4d (thus image).
# this way, we only apply the transformation in the spatial domain
def resize(vid, size, interpolation='bilinear'):
    # NOTE: using bilinear interpolation because we don't work on minibatches
    # at this level
    scale = None
    if isinstance(size, int):
        scale = float(size) / min(vid.shape[-2:])
        size = None
    return torch.nn.functional.interpolate(
        vid, size=size, scale_factor=scale, mode=interpolation, align_corners=False)


def pad(vid, padding, fill=0, padding_mode="constant"):
    # NOTE: don't want to pad on temporal dimension, so let as non-batch
    # (4d) before padding. This works as expected
    return torch.nn.functional.pad(vid, padding, value=fill, mode=padding_mode)


def to_normalized_float_tensor(vid):
    return vid.permute(3, 0, 1, 2).to(torch.float32) / 255


def normalize(vid, mean, std):
    shape = (-1,) + (1,) * (vid.dim() - 1)
    mean = torch.as_tensor(mean).reshape(shape)
    std = torch.as_tensor(std).reshape(shape)
    return (vid - mean) / std


# Class interface

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    @staticmethod
    def get_params(vid, output_size):
        """Get parameters for ``crop`` for a random crop.
        """
        h, w = vid.shape[-2:]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, vid):
        i, j, h, w = self.get_params(vid, self.size)
        return crop(vid, i, j, h, w)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return center_crop(vid, self.size)


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return resize(vid, self.size)
    

class RandomResizedCrop(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation='bilinear'):
        self.size = size
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    def __call__(self, vid):
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            vid, scale=self.scale, ratio=self.ratio)
        vid = crop(vid, i, j, h, w)
        vid = resize(vid, self.size, self.interpolation)
        return vid


class ToFloatTensorInZeroOne(object):
    def __call__(self, vid):
        return to_normalized_float_tensor(vid)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, vid):
        return normalize(vid, self.mean, self.std)


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, vid):
        if random.random() < self.p:
            return hflip(vid)
        return vid


class Pad(object):
    def __init__(self, padding, fill=0):
        self.padding = padding
        self.fill = fill

    def __call__(self, vid):
        return pad(vid, self.padding, self.fill)
    
class Solarize(object):
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, vid):
        return torch.where(vid < self.threshold, vid, 255 - vid)
    
class GaussianBlur(object):
    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, vid):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return torchvision.transforms.functional.gaussian_blur(vid, self.kernel_size, sigma)
    
class VideoColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
    
    def __call__(self, video_tensor):
        video_tensor = video_tensor.permute(1, 0, 2, 3)
        video_tensor_jittered = torch.empty_like(video_tensor)
        for i, frame in enumerate(video_tensor):
            # Convert tensor to PIL Image
            frame = transforms.ToPILImage()(frame)
            # Apply color jitter
            jittered_frame = self.color_jitter(frame)
            # Convert PIL Image back to tensor
            jittered_frame = transforms.ToTensor()(jittered_frame)
            # Place the transformed frame back into the video tensor
            video_tensor_jittered[i] = jittered_frame
        video_tensor_jittered = video_tensor_jittered.permute(1, 0, 2, 3)
        return video_tensor_jittered
    
class GrayScale(object):
    def __call__(self, vid):
        return vid.mean(dim=0, keepdim=True).repeat(3, 1, 1, 1)
    
# temporal transforms
class TemporalRandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        t = vid.shape[0]
        tt = self.size
        if t == tt:
            return vid
        elif t > tt:
            i = random.randint(0, t - tt)
            return vid[i:(i + tt)]
        else:
            raise ValueError("video is too short to perform this transform")