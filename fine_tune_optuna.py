import logging
import os
import optuna
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from utils.evaluate import evaluate
from utils.get_model import get_model
from utils.init_experiment import init_experiment
from utils.save_model_weights import save_best_model_weights


def objective(trial, config):
    # Generate the model
    model = get_model(config.checkpoint_restore_path)
    
    # freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze the conv1, blocks
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Generate the optimizer
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimz = optim.Adam(model.parameters(), lr)

    train_loader = config.train_loader
    valid_loader_small = config.valid_loader_small
    valid_loader = config.valid_loader

    N_EPOCHS = 1
    evaluation_counter = 0

    best_acc = 0.0
    for epoch in range(1, N_EPOCHS + 1):
        logging.info(f"Epoch: {epoch}/{N_EPOCHS}")
        
        for i, (videos, label) in tqdm(enumerate(train_loader)):
            model.train()
            model.clean_activation_buffers()
            optimz.zero_grad()

            # forward pass
            preds, _ = model(videos.cuda())

            # compute loss
            out = F.log_softmax(preds, dim=1)
            loss = F.nll_loss(out, label.cuda())

            # backward pass
            loss.backward()
            optimz.step()

            # print statistics
            if i % 50 == 0:
                accuracy = evaluate(model, valid_loader_small)
                logging.info(f"[Epoch {epoch}/{N_EPOCHS}] [Iteration {i}/{len(train_loader)}]  Accuracy: {accuracy} Latest Loss: Loss: {loss.item()}")
                trial.report(accuracy, evaluation_counter)
                evaluation_counter += 1

                # store model
                model_save_path = os.path.join(config.checkpoint_path, f"model_{trial.number}.pth")
                best_acc = save_best_model_weights(model, accuracy, best_acc, model_save_path)

                # prune
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
        
        # final evaluation
        accuracy = evaluate(model, valid_loader)

        

        return accuracy
      
    
if __name__ == "__main__":
    from functools import partial
    import configs.config as config

    init_experiment(config)

    logging.info(f"Start experiment: {config.experiment_name}")
    
    objective = partial(objective, config=config)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
        