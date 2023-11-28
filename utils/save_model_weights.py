import torch
import os

def save_model_weights(model,overall_accuracy, best_acc, base_path, file_name):
    best_weight = os.path.join(base_path, file_name)
    if overall_accuracy >= best_acc:
        best_acc = overall_accuracy
        print(f'New best accuracy. Save best weight to {best_weight}')
        torch.save(model.state_dict(), best_weight, _use_new_zipfile_serialization=False)
    return best_acc