import torch


def save_best_model_weights(model,current_accuracy, best_acc, save_path):
    if current_accuracy >= best_acc:
        best_acc = current_accuracy
        torch.save(model.state_dict(), save_path)
    return best_acc