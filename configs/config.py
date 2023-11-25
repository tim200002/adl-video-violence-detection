
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import utils.transforms as T
import utils.dataloader as dataloader

torch.manual_seed(97)
num_frames = 16 # 16
clip_steps = 1
Bs_Train = 16
Bs_Test = 16
checkpoint_path ='./checkpoint/'
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

train_path = './data/HockeyFights/train/'
valid_path = './data/HockeyFights/valid/'
test_path = './data/HockeyFights/test/'

print(train_path)
train_dataset = dataloader.Hockey(root=train_path, annotation_path='test_train_splits/', frames_per_clip=num_frames,step_between_clips = clip_steps,  train=0,transform=transform, num_workers=16)
valid_dataset = dataloader.Hockey(root=valid_path, annotation_path='test_train_splits/', frames_per_clip=num_frames,step_between_clips = clip_steps,  train=0,transform=transform_test, num_workers=16)
test_dataset = dataloader.Hockey(root=test_path, annotation_path='test_train_splits/', frames_per_clip=num_frames,step_between_clips = clip_steps,  train=0,transform=transform_test, num_workers=16)

train_loader = DataLoader(train_dataset, batch_size=Bs_Train, shuffle=True)
valid_loader  = DataLoader(valid_dataset, batch_size=Bs_Test, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=Bs_Test, shuffle=False)