import os
import time
import torch.nn.functional as F
import torch.optim as optim
import torch
from utils.evaluate import evaluate
from utils.exponential_decy import exponential_decay
from utils.save_model_weights import save_model_weights

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
        best_acc_val = save_model_weights(model, accuracy_val, best_acc_val, config.checkpoint_path, config.checkpoint_name)

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

if __name__ == "__main__":
    from movinets import MoViNet
    from movinets.config import _C
    from run_finetune import train_test_run

    import configs.config as config
    
    N_EPOCHS = 5
    model = MoViNet(_C.MODEL.MoViNetA3, causal = False, pretrained = True )

    # freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze the conv1, blocks
    for param in model.classifier.parameters():
        param.requires_grad = True

    train_test_run(model, config.train_loader, config.valid_loader, config.test_loader, config, N_EPOCHS=N_EPOCHS, lr=0.00005)
