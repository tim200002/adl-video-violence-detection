import logging
import math
import os
import torch
from tqdm import tqdm
from utils.evaluate import evaluate
from utils.get_model import get_model
import torch.optim as optim
from utils.init_experiment import init_experiment
from utils.sampler import InfinityDomainSampler
from utils.save_model_weights import save_best_model_weights
import torch.nn.functional as F

from utils.wasserstein_loss import WassersteinLoss


def train(config):
    # Genereate te model
    model = get_model(config.checkpoint_restore_path)

    # Generate the optimizer
    lr = config.lr
    optimz = optim.Adam(model.parameters(), lr)

    # Generate domain alignment loss
    domain_alignment_loss = WassersteinLoss()

    # Furhter hyperparameters
    mmd_weighting_factor = config.mmd_weighting_factor

    src_sampler = InfinityDomainSampler(config.train_loader_hockey)
    target_sampler = InfinityDomainSampler(config.train_loader_ucf)
    target_test_loader = config.valid_loader_ucf_small

    no_of_epoch = math.ceil(config.max_iterations / config.evaluate_every_iteration)
    itertation_counter = 0

    best_acc = 0.0

    for epoch in range(1, no_of_epoch + 1):
        logging.info(f"Epoch: {epoch}/{no_of_epoch}")
        iterations_this_epoch = min(config.evaluate_every_iteration, config.max_iterations - itertation_counter)
        for _ in tqdm(range(0, iterations_this_epoch)):
            itertation_counter += 1

            (src_videos, label) = src_sampler.get_sample()
            (target_videos, _) = target_sampler.get_sample()

            model.train()
            optimz.zero_grad()
            model.clean_activation_buffers()

            # forward pass
            preds, src_features = model(src_videos.cuda())
            _, target_features = model(target_videos.cuda())

            # Optimize for source domain accuracy
            out = F.log_softmax(preds, dim=1)
            loss_acc = F.nll_loss(out, label.cuda())

            # Optimize for domain alignment
            
            # Mean pool accross time
            if config.mean_pooling:
                src_features_domain_alignment = torch.mean(src_features, dim=2)
                target_features_domain_alignment = torch.mean(target_features, dim=2)
            else:
                src_features_domain_alignment = src_features
                target_features_domain_alignment = target_features

            # flatten
            src_feature_flattened = src_features_domain_alignment.view(src_features_domain_alignment.shape[0], -1)
            target_feature_flattened = target_features_domain_alignment.view(target_features_domain_alignment.shape[0], -1)

            loss_alignment = domain_alignment_loss(src_feature_flattened, target_feature_flattened)

            # Overall loss is calculated as: (1-mmd_weighting_factor)*classification_loss + mmd_weighting_factor*mmd_loss
            loss = (1-mmd_weighting_factor)*loss_acc + mmd_weighting_factor*loss_alignment
            loss.backward()
            optimz.step()
        
        # evaluate
        target_accuracy = evaluate(model, target_test_loader)
        logging.info(f"[Epoch: {epoch}/{no_of_epoch}, Iteration: {itertation_counter}/config.max_iterations] Target accuracy: {target_accuracy.item()}")
        model_save_path = os.path.join(config.checkpoint_path, f"domain_alignment.pth")
        best_acc = save_best_model_weights(model, target_accuracy, best_acc, model_save_path)
    
    return best_acc

if __name__ == "__main__":
    import configs.config_mmd as config
    init_experiment(config)

    # initial evaluation
    target_accuracy = evaluate(get_model(config.checkpoint_restore_path), config.valid_loader_ucf_small)
    logging.info(f"Initial accuracy: {target_accuracy}")

    best_acc = train(config)
    logging.info(f"Experiment finished. Best accuracy: {best_acc}")

    
    