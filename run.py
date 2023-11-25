import os
import time
import torch.nn.functional as F
import torch.optim as optim
import torch

import config


def save_model_weights(model,overall_accuracy, best_acc, path):
    best_weight = os.path.join(path, 'best.pt')
    if overall_accuracy >= best_acc:
        best_acc = overall_accuracy
        print(f'New best accuracy. Save best weight to {best_weight}')
        torch.save(model.state_dict(), best_weight, _use_new_zipfile_serialization=False)
    return best_acc


def train_iter(model, optimz, data_load, loss_history):
    samples = len(data_load.dataset)
    model.train()
    model.cuda()
    model.clean_activation_buffers()
    optimz.zero_grad()
    for i, (data, target) in enumerate(data_load):
        out = F.log_softmax(model(data.cuda()), dim=1)
        loss = F.nll_loss(out, target.cuda())
        loss.backward()
        optimz.step()
        optimz.zero_grad()
        model.clean_activation_buffers()
        if i % 50 == 0:
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_load)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            loss_history.append(loss.item())


def evaluate(model, data_load, loss_history):
    model.eval()

    samples = len(data_load.dataset)
    accuracy = 0
    tloss = 0
    model.clean_activation_buffers()
    with torch.no_grad():
        for data,  target in data_load:
            output = F.log_softmax(model(data.cuda()), dim=1)
            loss = F.nll_loss(output, target.cuda(), reduction='sum')
            _, pred = torch.max(output, dim=1)

            tloss += loss.item()
            accuracy += pred.eq(target.cuda()).sum()
            model.clean_activation_buffers()
    aloss = tloss / samples
    loss_history.append(aloss)
    print('\nAverage loss: ' + '{:.4f}'.format(aloss) +
          '  Accuracy:' + '{:5}'.format(accuracy) + '/' +
          '{:5}'.format(samples) + ' (' +
          '{:4.2f}'.format(100.0 * accuracy / samples) + '%)\n')
    

    return accuracy


def exponential_decay(optimizer, epoch, decay_rate=0.96, lr_decay_epoch=2):
    """Decay learning rate by a factor of decay_rate every lr_decay_epoch epochs."""
    if epoch == 0:
        return optimizer
    
    if epoch % lr_decay_epoch:
        return optimizer

    for param_group in optimizer.param_groups:
        print("Exponential Decay updating learning rate")
        param_group['lr'] *= decay_rate

    return optimizer


def train_test_run(model, train_load, val_load, test_load, config,  N_EPOCHS=5, lr=0.00005):
    start_time = time.time()
    losses_train, losses_val, losses_test = [], [], []

    optimz = optim.Adam(model.parameters(), lr=lr)

    best_acc_val = 0

    for epoch in range(0, N_EPOCHS):
        print('Epoch:', epoch)
        optimz = exponential_decay(optimz, epoch)
        
        print('Started Training -------------------------------')
        train_iter(model, optimz, train_load, losses_train)

        print('Started Validation -----------------------------')
        accuracy_val = evaluate(model, val_load, losses_val)
        best_acc_val = save_model_weights(model, accuracy_val, best_acc_val, config.checkpoint_path)

        print('Started Testing --------------------------------')
        accuracy_test = evaluate(model, test_load, losses_test)
         
    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
    return {
        'train_loss': losses_train,
        'val_loss': losses_val,
        'test_loss': losses_test,
        'best_acc_val': best_acc_val,
        'accuracy_test': accuracy_test,
        'accuracy_val': accuracy_val,
    }
