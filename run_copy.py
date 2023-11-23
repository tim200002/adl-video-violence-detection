import os
import time
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import torch
import transforms as T
from movinets import MoViNet
from movinets.config import _C

import config
import movinets.models

#sometimes = lambda aug: va.Sometimes(0.1, aug) # Used to apply augmentor with 10% probability
#seq = va.OneOF([
#    #va.RandomCrop(size=(240, 180)), # randomly crop video with a size of (240 x 180)
#    #va.RandomRotate(degrees=10), # randomly rotates the video with a degree randomly choosen from [-10, 10]
#    sometimes(va.GaussianBlur()),
#    sometimes(va.ElasticTransformation()),
#    sometimes(va.PiecewiseAffineTransform()),
#    sometimes(va.Superpixel()),
#    sometimes(va.Pepper()),
#    sometimes(va.Salt()),
#    sometimes(va.InvertColor())
#])

def save_model_weights(model,overall_accuracy, best_acc):
    best_weight = os.path.join(config.checkpoint_path, 'best.pt')
    if overall_accuracy >= best_acc:
        best_acc = overall_accuracy
        print('Save Weight!',best_weight)
        torch.save(model.state_dict(), best_weight, _use_new_zipfile_serialization=False)
    return best_acc

def train_iter(model, optimz, data_load, loss_val):
    samples = len(data_load.dataset)
    model.train()
    model.cuda()
    model.clean_activation_buffers()
    optimz.zero_grad()
    for i, (data, target) in enumerate(data_load):
        out = F.log_softmax(model(data.cuda()), dim=1)
        loss = F.nll_loss(out, target.cuda())
        loss.backward()
        optimz.step()
        optimz.zero_grad()
        model.clean_activation_buffers()
        if i % 50 == 0:
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_load)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            loss_val.append(loss.item())

def evaluate(model, data_load, loss_val, best_acc):
    model.eval()

    samples = len(data_load.dataset)
    csamp = 0
    tloss = 0
    model.clean_activation_buffers()
    with torch.no_grad():
        for data,  target in data_load:
            output = F.log_softmax(model(data.cuda()), dim=1)
            loss = F.nll_loss(output, target.cuda(), reduction='sum')
            _, pred = torch.max(output, dim=1)

            tloss += loss.item()
            csamp += pred.eq(target.cuda()).sum()
            model.clean_activation_buffers()
    aloss = tloss / samples
    loss_val.append(aloss)
    print('\nAverage test loss: ' + '{:.4f}'.format(aloss) +
          '  Accuracy:' + '{:5}'.format(csamp) + '/' +
          '{:5}'.format(samples) + ' (' +
          '{:4.2f}'.format(100.0 * csamp / samples) + '%)\n')
    best_acc = save_model_weights(model, csamp, best_acc)

    return best_acc


def train_iter_stream(model, optimz, data_load, loss_val, n_clips = 2, n_clip_frames=8):
    """
    In causal mode with stream buffer a single video is fed to the network
    using subclips of lenght n_clip_frames.
    n_clips*n_clip_frames should be equal to the total number of frames presents
    in the video.

    n_clips : number of clips that are used
    n_clip_frames : number of frame contained in each clip
    """
    #clean the buffer of activations
    samples = len(data_load.dataset)
    model.cuda()
    model.train()
    model.clean_activation_buffers()
    optimz.zero_grad()
    csamp = 0
    print("best",best_acc)
    for i, (data, target) in enumerate(data_load):
        data = data.cuda()
        #data = seq(data)
        #print("==",data.size())
        target = target.cuda()
        #print(target.size())
        l_batch = 0
        #backward pass for each clip
        for j in range(n_clips):
          #print(data.size())
          #print(data[:,:,(n_clip_frames)*(j):(n_clip_frames)*(j+1)].size())
          output = F.log_softmax(model(data[:,:,(n_clip_frames)*(j):(n_clip_frames)*(j+1)]), dim=1)
          #print(data[:,:,(n_clip_frames)*(j):(n_clip_frames)*(j+1)])
          loss = F.nll_loss(output, target)
          _, pred = torch.max(output, dim=1)
          #print("target",target)
          #print("output",pred)
          loss = F.nll_loss(output, target)/n_clips
          loss.backward()
        _, pred = torch.max(output, dim=1)
        csamp += pred.eq(target).sum()
        
        l_batch += loss.item()*n_clips
        optimz.step()
        optimz.zero_grad()

        #clean the buffer of activations
        model.clean_activation_buffers()
        #break
        if i % 50 == 0:
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_load)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(l_batch))
            loss_val.append(l_batch)
    #save_model_weights(csamp, best_acc)

def evaluate_stream(model, data_load, loss_val, best_acc, n_clips = 2, n_clip_frames=8):
    model.eval()
    model.cuda()
    samples = len(data_load.dataset)
    csamp = 0
    tloss = 0
    with torch.no_grad():
        for data, target in data_load:
            data = data.cuda()
            target = target.cuda()
            model.clean_activation_buffers()
            for j in range(n_clips):
              output = F.log_softmax(model(data[:,:,(n_clip_frames)*(j):(n_clip_frames)*(j+1)]), dim=1)
              loss = F.nll_loss(output, target)
            _, pred = torch.max(output, dim=1)
            tloss += loss.item()
            csamp += pred.eq(target).sum()

    aloss = tloss /  len(data_load)
    loss_val.append(aloss)
    print('\nAverage test loss: ' + '{:.4f}'.format(aloss) +
          '  Accuracy:' + '{:5}'.format(csamp) + '/' +
          '{:5}'.format(samples) + ' (' +
          '{:4.2f}'.format(100.0 * csamp / samples) + '%)\n')
    best_acc = save_model_weights(model, csamp, best_acc)
        
    return best_acc

if __name__ == "__main__":
    # cuda visibile devices
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    N_EPOCHS = 5
    model = MoViNet(_C.MODEL.MoViNetA3, causal = False, pretrained = True )
    
    # freeze all layers except the blocks layer
    for param in model.parameters():
        param.requires_grad = False

    start_time = time.time()

    trloss_val, tsloss_val = [], []
    model.classifier[3] = torch.nn.Conv3d(2048, 2, (1,1,1))
    optimz = optim.Adam(model.parameters(), lr=0.00005)
    best_acc=0
    best_acc = evaluate_stream(model, config.valid_loader, tsloss_val, best_acc=best_acc)

    for epoch in range(1, N_EPOCHS + 1):
        print('Epoch:', epoch)
        train_iter(model, optimz, config.train_loader, trloss_val)
        best_acc = evaluate(model, config.valid_loader, tsloss_val, best_acc=best_acc)
        if epoch > 0 and (epoch  % 2 == 0):
                for param_group in optimz.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5
                    print("lr", param_group['lr'])

    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')