"""
    Self-Supervised Learning via Maximum Entropy Coding

    36th Conference on Neural Information Processing Systems (NeurIPS 2022)

    The code is mainly based on https://github.com/facebookresearch/simsiam
"""

import os

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import numpy as np

import mec.loader
import mec.builder

import logging
import math
import os
from tqdm import tqdm
from utils.evaluate import evaluate
from utils.get_model import get_model
import torch.optim as optim
from utils.init_experiment import init_experiment
from utils.sampler import InfinityDomainSampler
from utils.save_model_weights import save_best_model_weights


def train(config):
    # Genereate te model
    model = get_model(model_name="A1", checkpoint_path=config.checkpoint_restore_path)
    model = mec.builder.MEC(model, dim=config.dim)
    model.cuda()

    # Generate the optimizer
    lr = config.lr
    optimz = optim.Adam(model.parameters(), lr)

    src_sampler = InfinityDomainSampler(config.train_loader_hockey, config.Bs_Train)
    target_sampler = InfinityDomainSampler(config.train_loader_ucf, config.Bs_Train)
    target_test_loader = config.valid_loader_ucf_small

    no_of_epoch = math.ceil(config.max_iterations / config.evaluate_every_iteration)
    itertation_counter = 0

    best_acc = 0.0

    mec_loss = config.mec_loss
    eps_d = config.eps / config.dim
    lamda = 1 / (config.Bs_Train * eps_d)
    lamda_schedule = lamda_scheduler(8/lamda, 1/lamda, no_of_epoch, config.evaluate_every_iteration, warmup_epochs=config.warmpup_epochs)
    momentum_schedule = cosine_scheduler(config.teacher_momentum, 1, no_of_epoch, config.evaluate_every_iteration)

    for epoch in range(1, no_of_epoch + 1):
        logging.info(f"Epoch: {epoch}/{no_of_epoch}")
        iterations_this_epoch = min(config.evaluate_every_iteration, config.max_iterations - itertation_counter)
        for _ in tqdm(range(0, iterations_this_epoch)):
            model.train()
            optimz.zero_grad()
            model.encoder.clean_activation_buffers()

            (src_videos, label) = src_sampler.get_sample()
            (target_videos1, target_videos2, _) = target_sampler.get_sample()

            lamda_inv = lamda_schedule[itertation_counter]
            momentum = momentum_schedule[itertation_counter]

            itertation_counter += 1

            # forward pass
            preds, _ = model.encoder(src_videos.cuda())

            # compute loss
            out = torch.log_softmax(preds, dim=1)
            loss_acc = torch.nn.functional.nll_loss(out, label.cuda())

            # compute output and loss
            z1, z2, p1, p2, _, _ = model(target_videos1.cuda(), target_videos2.cuda())

            ssl_loss = mec_loss(p1, p2, z1, z2, lamda_inv)

            # scaled loss by lamda
            loss = loss_acc - config.mec_weighting_factor * lamda_inv * ssl_loss

            # compute gradient and do SGD step
            optimz.zero_grad()
            loss.backward()
            optimz.step()

            # momentum update of the parameters of the teacher network
            with torch.no_grad():
                for param_q, param_k in zip(model.encoder.parameters(), model.teacher.parameters()):
                    param_k.data.mul_(momentum).add_((1 - momentum) * param_q.detach().data)

        # evaluate
        target_accuracy, confusion_matrix = evaluate(model.encoder, target_test_loader)
        logging.info(f"[Epoch: {epoch}/{no_of_epoch}, Iteration: {itertation_counter}/{config.max_iterations}] Target accuracy: {target_accuracy} Confusion matrix: {confusion_matrix}")
        model_save_path = os.path.join(config.checkpoint_path, f"ssl.pth")
        best_acc = save_best_model_weights(model.encoder, target_accuracy, best_acc, model_save_path)
    return best_acc


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def lamda_scheduler(start_warmup_value, base_value, epochs, niter_per_ep, warmup_epochs=5):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    schedule = np.ones(epochs * niter_per_ep - warmup_iters) * base_value
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


if __name__ == '__main__':
    import configs.config_mec as config
    init_experiment(config)

    # initial evaluation
    model = get_model(model_name="A1", checkpoint_path=config.checkpoint_restore_path)
    target_accuracy, evaluation_matrix = evaluate(model=model, data_load=config.valid_loader_ucf_small)
    logging.info(f"Initial accuracy: {target_accuracy}")
    logging.info(f"Initial evaluation matrix: {evaluation_matrix}")

    best_acc = train(config)
    logging.info(f"Experiment finished. Best accuracy: {best_acc}")
