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

class HMDB51_to_ViolenceWrapper(torch.utils.data.Dataset):
    def __init__(self, directory, train, clip_steps, transform, num_frames):
        self.dataset = torchvision.datasets.HMDB51(directory, 'test_train_splits/', num_frames, frame_rate=5,
                                                step_between_clips = clip_steps, fold=1, train=train,
                                                transform=transform, num_workers=2)
        self.violence_classes_names = ['fencing', 'hit', 'kick', 'punch', 'shoot_bow', 'shoot_gun', 'sword', 'sword_exercise']
        self.violence_classes = [self.dataset.classes.index(name) for name in self.violence_classes_names]
        
    def __getitem__(self, index):
        data, _, target = self.dataset[index]
        # target 0 if not violence, 1 if violence
        target = torch.tensor(1 if target in self.violence_classes else 0)
        return data, target
    
    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    torch.manual_seed(97)
    num_frames = 16 # 16
    clip_steps = 2
    Bs_Train = 16
    Bs_Test = 16

    transform = transforms.Compose([  
                                                        
                                    T.ToFloatTensorInZeroOne(),
                                    T.Resize((200, 200)),
                                    T.RandomHorizontalFlip(),
                                    #T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
                                    T.RandomCrop((172, 172))])
    transform_test = transforms.Compose([                           
                                    T.ToFloatTensorInZeroOne(),
                                    T.Resize((200, 200)),
                                    #T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
                                    T.CenterCrop((172, 172))])

    hmdb51_train = HMDB51_to_ViolenceWrapper('video_data/', train=True, clip_steps=clip_steps, transform=transform, num_frames=num_frames)

    hmdb51_test = HMDB51_to_ViolenceWrapper('video_data/', train=False, clip_steps=clip_steps, transform=transform_test, num_frames=num_frames)

    train_loader = DataLoader(hmdb51_train, batch_size=Bs_Train, shuffle=True)
    test_loader  = DataLoader(hmdb51_test, batch_size=Bs_Test, shuffle=False)