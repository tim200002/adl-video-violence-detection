import argparse
import utils.dataloader as dataloader
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from utils.evaluate import evaluate
import utils.transforms as T
import torch

from utils.get_model import get_model

def prepare_data():
    transform_test = transforms.Compose([
                                    T.ToFloatTensorInZeroOne(),
                                    T.Resize((200, 200)),
                                    #T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
                                    T.CenterCrop((172, 172))])
    num_frames = 16 # 16
    frame_rate = 5
    clip_steps = 1
    Bs_Train = 10
    Bs_Test = 16

    valid_path_hockey = './data/HockeyFights/valid/'
    test_path_hockey = './data/HockeyFights/test/'

    train_path_ucf = './data/ucf-fighting-split/train/'
    valid_path_ucf = './data/ucf-fighting-split/val/'
    test_path_ucf = './data/ucf-fighting-split/test/'


    
    valid_dataset_hockey = dataloader.Hockey(root=valid_path_hockey, frames_per_clip=num_frames,step_between_clips = clip_steps,  train=0,transform=transform_test, num_workers=16)
    test_dataset_hockey = dataloader.Hockey(root=test_path_hockey,  frames_per_clip=num_frames,step_between_clips = clip_steps,  train=0,transform=transform_test, num_workers=16)

    train_dataset_ucf  = dataloader.UCF(root=train_path_ucf, frames_per_clip=num_frames,frame_rate=frame_rate,step_between_clips = clip_steps, train=0,transform=transform_test, num_workers=8) 
    train_dataset_ucf_small  = torch.utils.data.Subset(train_dataset_ucf, torch.randperm(len(train_dataset_ucf))[:500])
    valid_dataset_ucf = dataloader.UCF(root=valid_path_ucf, frames_per_clip=num_frames,frame_rate=frame_rate,step_between_clips = clip_steps, train=0,transform=transform_test, num_workers=8)
    test_dataset_ucf = dataloader.UCF(root=test_path_ucf, frames_per_clip=num_frames,frame_rate=frame_rate,step_between_clips = clip_steps,  train=0,transform=transform_test, num_workers=8)

    valid_loader_hockey  = DataLoader(valid_dataset_hockey, batch_size=Bs_Test, shuffle=False)
    test_loader_hockey  = DataLoader(test_dataset_hockey, batch_size=Bs_Test, shuffle=False)
    
    train_loader_ucf  = DataLoader(train_dataset_ucf_small, batch_size=Bs_Train, shuffle=True)
    valid_loader_ucf  = DataLoader(valid_dataset_ucf, batch_size=Bs_Test, shuffle=False)
    test_loader_ucf  = DataLoader(test_dataset_ucf, batch_size=Bs_Test, shuffle=False)

    return {
        "src_valid_loader": valid_loader_hockey,
        "src_test_loader": test_loader_hockey,
        "tgt_train_loader": train_loader_ucf,
        "tgt_valid_loader": valid_loader_ucf,
        "tgt_test_loader": test_loader_ucf
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate source and target domain')
    parser.add_argument('--checkpoint', type=str, help='path to checkpoint')
    parser.add_argument('--model', type=str, default='A1', help='model name')

    args = parser.parse_args()
    

    model = get_model(model_name=args.model, checkpoint_path=args.checkpoint)
    data = prepare_data()

    print("Evaluating source domain----------------------")
    print("Validation Set")
    accuracy, evauation_matrix = evaluate(model, data["src_valid_loader"])
    print(f"Accuracy: {accuracy}, Confusion Matrix: {evauation_matrix}")

    print("Test Set")
    accuracy, evauation_matrix = evaluate(model, data["src_test_loader"])
    print(f"Accuracy: {accuracy}, Confusion Matrix: {evauation_matrix}")

    print("Evaluating target domain----------------------")

    print("Train Set")
    accuracy, evauation_matrix = evaluate(model, data["tgt_train_loader"])
    print(f"Accuracy: {accuracy}, Confusion Matrix: {evauation_matrix}")

    print("Validation Set")
    accuracy, evauation_matrix = evaluate(model, data["tgt_valid_loader"])
    print(f"Accuracy: {accuracy}, Confusion Matrix: {evauation_matrix}")

    print("Test Set")
    accuracy, evauation_matrix = evaluate(model, data["tgt_test_loader"])
    print(f"Accuracy: {accuracy}, Confusion Matrix: {evauation_matrix}")



    