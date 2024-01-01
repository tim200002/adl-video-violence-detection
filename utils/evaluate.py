import torch
from tqdm import tqdm
import torch.nn.functional as F


def evaluate(model, data_load, config):
    model.eval()
    # Matrix in the format
    # [[True Non Violence, False Non Violence],
    #  [False  Violence, True  Violence]]
    counting_matrix = [[0,0],[0,0]]
    acc = 0.0
    n_subclip_frames = config.num_frames // config.n_subclips

    with torch.no_grad():
        for videos,  labels in tqdm(data_load):
            model.clean_activation_buffers()

            for j in range(config.n_subclips):
                output, _ = model(videos.cuda()[:,:,(n_subclip_frames)*(j):(n_subclip_frames)*(j+1)])

            output = torch.sigmoid(output)
            pred = output > 0.5
            for (label, prediction) in zip(labels, pred):
                counting_matrix[label][prediction] += 1

            acc += torch.sum(pred == labels.cuda()).item()
    
    # Normalize the matrix
    no_true_violent_samples = counting_matrix[0][0] + counting_matrix[0][1]
    no_true_non_violent_samples = counting_matrix[1][0] + counting_matrix[1][1]

    counting_matrix_normalized = [[0,0],[0,0]]

    counting_matrix_normalized[0][0] = counting_matrix[0][0] / no_true_violent_samples
    counting_matrix_normalized[0][1] = counting_matrix[0][1] / no_true_violent_samples
    counting_matrix_normalized[1][0] =  counting_matrix[1][0] / no_true_non_violent_samples
    counting_matrix_normalized[1][1] = counting_matrix[1][1] / no_true_non_violent_samples      
    
    normalized_accuracy = (counting_matrix_normalized[0][0] + counting_matrix_normalized[1][1]) / 2

    return normalized_accuracy, counting_matrix_normalized

            