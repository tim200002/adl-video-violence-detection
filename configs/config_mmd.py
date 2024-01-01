
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from utils.mmd_loss import RBF, MMDLoss
from utils.mec_loss import MECLoss
import utils.transforms as T
import utils.dataloader as dataloader
from utils.sampler import InfinityDomainSampler
from utils.wasserstein_loss import WassersteinLoss

torch.manual_seed(97)
experiment_name = "domain_alignment"
num_frames = 16 # 16
frame_rate = 5
clip_steps = 1
Bs_Train = 10
Bs_Test = 16

checkpoint_restore_path = "checkpoint/best_hockey_A1.pt"

# parameters, automatically set by optuna, set manually for manual runs
lr = 0.0006949557944820544
mmd_weighting_factor = 0.567610878889856

# special mmd parameters
max_iterations = 2500
evaluate_every_iteration = 50

kernel = RBF(device="cuda")
mec_loss = MECLoss()
# domain_alignment_loss = MMDLoss(kernel=kernel)
domain_alignment_loss = WassersteinLoss()

should_augment = False


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

train_path_hockey = './data/HockeyFights/train/'
valid_path_hockey = './data/HockeyFights/valid/'
test_path_hockey = './data/HockeyFights/test/'

train_path_ucf = './data/ucf-fighting-split/train/'
valid_path_ucf = './data/ucf-fighting-split/val/'
test_path_ucf = './data/ucf-fighting-split/test/'

train_dataset_hockey = dataloader.Hockey(root=train_path_hockey,  frames_per_clip=num_frames,step_between_clips = clip_steps,  train=0,transform=transform, num_workers=16)
valid_dataset_hockey = dataloader.Hockey(root=valid_path_hockey, frames_per_clip=num_frames,step_between_clips = clip_steps,  train=0,transform=transform_test, num_workers=16)
test_dataset_hockey = dataloader.Hockey(root=test_path_hockey,  frames_per_clip=num_frames,step_between_clips = clip_steps,  train=0,transform=transform_test, num_workers=16)

train_loader_hockey = DataLoader(train_dataset_hockey, batch_size=Bs_Train, shuffle=True)
valid_loader_hockey  = DataLoader(valid_dataset_hockey, batch_size=Bs_Test, shuffle=False)
test_loader_hockey  = DataLoader(test_dataset_hockey, batch_size=Bs_Test, shuffle=False)

train_dataset_ucf = dataloader.UCF(root=train_path_ucf, frames_per_clip=num_frames,frame_rate=frame_rate,step_between_clips = clip_steps,  train=0,transform=transform, num_workers=8)
valid_dataset_ucf = dataloader.UCF(root=valid_path_ucf, frames_per_clip=num_frames,frame_rate=frame_rate,step_between_clips = clip_steps, train=0,transform=transform_test, num_workers=8)
# reduce size to 100 random samples
valid_dataset_ucf_small = torch.utils.data.Subset(valid_dataset_ucf, torch.randperm(len(valid_dataset_ucf))[:100])

test_dataset_ucf = dataloader.UCF(root=test_path_ucf, frames_per_clip=num_frames,frame_rate=frame_rate,step_between_clips = clip_steps,  train=0,transform=transform_test, num_workers=8)

train_loader_ucf = DataLoader(train_dataset_ucf, batch_size=Bs_Train, shuffle=True)
valid_loader_ucf  = DataLoader(valid_dataset_ucf, batch_size=Bs_Test, shuffle=False)
valid_loader_ucf_small  = DataLoader(valid_dataset_ucf_small, batch_size=Bs_Test, shuffle=False)
test_loader_ucf  = DataLoader(test_dataset_ucf, batch_size=Bs_Test, shuffle=False)