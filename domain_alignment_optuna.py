import argparse
import logging
import math
import os
import torch.optim as optim
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.augmentation import get_augmenter_2

from utils.evaluate import evaluate
from utils.get_model import get_model
from utils.init_experiment import init_experiment, resume_experiment
from utils.mmd_loss import RBF, MMDLoss
from utils.sampler import InfinityDomainSampler
from utils.save_model_weights import save_best_model_weights


from utils.wasserstein_loss import WassersteinLoss
import optuna

def objective(trial, config):
    # Generate the model
    model = get_model(model_name="A1", checkpoint_path=config.checkpoint_restore_path)
    # Generate the optimizer
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    optimz = optim.Adam(model.parameters(), lr)
    # exponential_decay_rate = trial.suggest_loguniform("exponential_decay_rate", 0.0001, 0.1)

    # Generate domain alignment loss
    domain_alignment_loss_string = trial.suggest_categorical("domain_alignment_loss", ["wasserstein", "mmd"])
    if domain_alignment_loss_string == "wasserstein":
        domain_alignment_loss = WassersteinLoss()
    elif domain_alignment_loss_string == "mmd":
        kernel = RBF(device="cuda")
        domain_alignment_loss = MMDLoss(kernel=kernel)
    else:
        raise NotImplementedError()

    # Furhter hyperparameters
    mmd_weighting_factor = trial.suggest_float("mmd_weighting_factor", 0.2, 0.8)
    mean_pooling = trial.suggest_categorical("mean_pooling", [True, False])

    #should_augment_video = trial.suggest_categorical("should_augment_video", [True, False])
    should_augment_video = False
    
    src_sampler = InfinityDomainSampler(config.train_loader_hockey, config.Bs_Train)
    target_sampler = InfinityDomainSampler(config.train_loader_ucf, config.Bs_Train)
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

            if should_augment_video:
                src_videos = get_augmenter_2(src_videos)
                target_videos = get_augmenter_2(target_videos)

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
            if mean_pooling:
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
        target_accuracy, confusion_matrix = evaluate(model, target_test_loader)
        logging.info(f"[Epoch: {epoch}/{no_of_epoch}, Iteration: {itertation_counter}/config.max_iterations] Target accuracy: {target_accuracy} Confusion matrix: {confusion_matrix}")
        trial.report(target_accuracy, epoch)

        model_save_path = os.path.join(config.checkpoint_path, f"model_{trial.number}.pth")
        best_acc = save_best_model_weights(model, target_accuracy, best_acc, model_save_path)

        # handle pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
    return best_acc

if __name__ == "__main__":
    from functools import partial

    
    # pass command line arguments for resume path
    parser = argparse.ArgumentParser()
    parser.add_argument("--study_restore_path", type=str, default=None, help="Path to study to resume")
    args = parser.parse_args()

    if args.study_restore_path is not None:
        resume_path = args.study_restore_path
        config = resume_experiment(resume_path)
    else:
        import configs.config_mmd as config
        init_experiment(config)
        # intial evaluation
        model = get_model(model_name="A1", checkpoint_path=config.checkpoint_restore_path)
        target_accuracy, confusion_matrix = evaluate(model, config.valid_loader_ucf_small)
        logging.info(f"Initial accuracy: {target_accuracy}")
        logging.info(f"Confusion matrix: {confusion_matrix}")
        
    
    # run study
    objective = partial(objective, config=config)
    
   
    study = optuna.create_study(direction="maximize", study_name=config.experiment_name, storage=config.study_path, load_if_exists=True)
    study.optimize(objective, n_trials=20)
