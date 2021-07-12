# MoViNet-pytorch
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Atze00/MoViNet-pytorch/blob/main/movinet_tutorial.ipynb)  [![Paper](http://img.shields.io/badge/Paper-arXiv.2103.11511-B3181B?logo=arXiv)](https://arxiv.org/abs/2103.11511) <br><br>
Pytorch unofficial implementation of [MoViNets: Mobile Video Networks for Efficient Video Recognition](https://arxiv.org/pdf/2103.11511.pdf). <br>
Authors: Dan Kondratyuk, Liangzhe Yuan, Yandong Li, Li Zhang, Mingxing Tan, Matthew Brown, Boqing Gong (Google Research) <br>
[[Authors' Implementation]](https://github.com/tensorflow/models/tree/master/official/vision/beta/projects/movinet)<br>

## Stream Buffer
![stream buffer](https://github.com/Atze00/MoViNet-pytorch/blob/main/figures/Stream_buffer.png)

#### Clean stream buffer
It is required to clean the buffer after all the clips of the same video have been processed.
```python
model.clean_activation_buffers()
```
## Usage
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Atze00/MoViNet-pytorch/blob/main/movinet_tutorial.ipynb) <br>
Click on "Open in Colab" to open an example of training on HMDB-51 <br> 
### installation 
```pip install git+https://github.com/Atze00/MoViNet-pytorch.git```

#### How to build a model
Use ```causal = True``` to use the model with stream buffer, causal = False will use standard convolutions<br>
```python
from movinets import MoViNet
from movinets.config import _C

number_classes = 600

MoViNetA0 = MoViNet(_C.MODEL.MoViNetA0, number_classes, causal = True)
MoViNetA1 = MoViNet(_C.MODEL.MoViNetA1, number_classes, causal = True)
...
```
##### Load weights
Use ```pretrained = True``` to use the model with pretrained weights, ```tf_like = True``` is also required.<br>

tf_like indicated that the model will behave like a tensorflow model in some restricted scenarios. <br>
tf_like is necessary in order to obtain models that work with tensorflow weights released by the autors. <br>
tf_like behaviour should not be used when you are trying to train a network from scratch, the functionalities are very limited and the speed of the network is slightly reduced.<br>

```python
MoViNetA2 = MoViNet(_C.MODEL.MoViNetA2, 600, causal = False, pretrained = True, tf_like = True )
```


#### Training loop examples
Training loop with stream buffer
```python
def train_iter(model, optimz, data_load, n_clips = 5, n_clip_frames=8):
    """
    In causal mode with stream buffer a single video is fed to the network
    using subclips of lenght n_clip_frames. 
    n_clips*n_clip_frames should be equal to the total number of frames presents
    in the video.
    
    n_clips : number of clips that are used
    n_clip_frames : number of frame contained in each clip
    """
    
    #clean the buffer of activations
    model.clean_activation_buffers()
    optimz.zero_grad()
    for i, data, target in enumerate(data_load):
        #backward pass for each clip
        for j in range(n_clips):
          out = F.log_softmax(model(data[:,:,(n_clip_frames)*(j):(n_clip_frames)*(j+1)]), dim=1)
          loss = F.nll_loss(out, target)/n_clips
          loss.backward()
        optimz.step()
        optimz.zero_grad()
        
        #clean the buffer of activations
        model.clean_activation_buffers()
```
Training loop with standard convolutions
```python
def train_iter(model, optimz, data_load):

    optimz.zero_grad()
    for i, (data,_ , target) in enumerate(data_load):
        out = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(out, target)
        loss.backward()
        optimz.step()
        optimz.zero_grad()
```

## Pretrained models
#### Weights
The weights are loaded from the tensorflow models released by the authors, trained on kinetics.

#### Base Models

Base models implement standard 3D convolutions without stream buffers.

| Model Name | Top-1 Accuracy* | Top-5 Accuracy* | Input Shape |
|------------|----------------|----------------|-------------|
| MoViNet-A0-Base | 72.28 | 90.92 | 50 x 172 x 172 | 
| MoViNet-A1-Base | 76.69 | 93.40 | 50 x 172 x 172 | 
| MoViNet-A2-Base | 78.62 | 94.17 | 50 x 224 x 224 | 
| MoViNet-A3-Base | 81.79 | 95.67 | 120 x 256 x 256 | 
| MoViNet-A4-Base | 83.48 | 96.16 | 80 x 290 x 290 | 
| MoViNet-A5-Base | 84.27 | 96.39 | 120 x 320 x 320 | 

*Accuracy reported on the official repository for the dataset kinetics 600, It has not been tested by me. It should be the same since the tf models and the reimplemented pytorch models output the same results [[Test]](https://github.com/Atze00/MoViNet-pytorch/blob/main/tests/test_pretrained_models.py).


#### Status
Currently are available the pretrained models for the following architectures:
- [x] MoViNetA1-BASE
- [ ] MoViNetA1-STREAM
- [x] MoViNetA2-BASE
- [ ] MoViNetA2-STREAM
- [x] MoViNetA3-BASE
- [ ] MoViNetA3-STREAM
- [x] MoViNetA4-BASE
- [ ] MoViNetA4-STREAM
- [x] MoViNetA5-BASE
- [ ] MoViNetA5-STREAM

### Citations
```bibtex
@article{kondratyuk2021movinets,
  title={MoViNets: Mobile Video Networks for Efficient Video Recognition},
  author={Dan Kondratyuk, Liangzhe Yuan, Yandong Li, Li Zhang, Matthew Brown, and Boqing Gong},
  journal={arXiv preprint arXiv:2103.11511},
  year={2021}
}
```
