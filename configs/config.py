
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import utils.transforms as T
import utils.dataloader as dataloader

torch.manual_seed(97)
experiment_name = "finetune"
num_frames = 16 # 16
clip_steps = 1
Bs_Train = 16
Bs_Test = 16

# parameters, automatically set by optuna, set manually for manual runs
lr = 0.00005
n_epochs = 1

checkpoint_restore_path = None

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

train_dataset_hockey = dataloader.Hockey(root=train_path, frames_per_clip=num_frames,step_between_clips = clip_steps,  train=0,transform=transform, num_workers=16)
valid_dataset_hockey = dataloader.Hockey(root=valid_path,  frames_per_clip=num_frames,step_between_clips = clip_steps,  train=0,transform=transform_test, num_workers=16)
valid_dataset_hockey_small = torch.utils.data.Subset(valid_dataset_hockey, torch.randperm(len(valid_dataset_hockey))[:100])
test_dataset_hockey = dataloader.Hockey(root=test_path,  frames_per_clip=num_frames,step_between_clips = clip_steps,  train=0,transform=transform_test, num_workers=16)

train_loader = DataLoader(train_dataset_hockey, batch_size=Bs_Train, shuffle=True)
valid_loader  = DataLoader(valid_dataset_hockey, batch_size=Bs_Test, shuffle=False)
valid_loader_small  = DataLoader(valid_dataset_hockey_small, batch_size=Bs_Test, shuffle=False)
test_loader  = DataLoader(test_dataset_hockey, batch_size=Bs_Test, shuffle=False)
