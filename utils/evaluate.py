import torch
from tqdm import tqdm
import torch.nn.functional as F


def evaluate(model, data_load):
    model.eval()

    samples = len(data_load.dataset)
    accuracy = 0

    with torch.no_grad():
        for videos,  labels in tqdm(data_load):
            model.clean_activation_buffers()

            output, _ = model(videos.cuda())
            output = F.log_softmax(output, dim=1)
            _, pred = torch.max(output, dim=1)
            accuracy += pred.eq(labels.cuda()).sum()

           
    accuracy_normalized = 100.0 * accuracy / samples
    return accuracy_normalized
            