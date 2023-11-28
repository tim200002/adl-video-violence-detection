import torch.optim as optim
import time
import torch

from utils.exponential_decy import exponential_decay
from utils.mmd_loss import RBF, MMDLoss
from utils.save_model_weights import save_model_weights
import torch.nn.functional as F


def evaluate(model, data_load):
    model.eval()

    samples = len(data_load.dataset)
    accuracy = 0

    with torch.no_grad():
        for data,  target in data_load:
            model.clean_activation_buffers()
            output = F.log_softmax(model(data.cuda()), dim=1)
            _, pred = torch.max(output, dim=1)
            accuracy += pred.eq(target.cuda()).sum()
    
    accuracy_normalized = 100.0 * accuracy / samples
    return accuracy_normalized
            

def train_iter(model, optimz, src_data_loader, target_sampler, mmd_weighting_factor, loss_history):
    samples = len(src_data_loader.dataset)
    model.train()
    model.cuda()

    kernel = RBF(device='cuda')
    mmd_loss = MMDLoss(kernel=kernel)
    
    for i, (src_videos, label) in enumerate(src_data_loader):
        optimz.zero_grad()
        model.clean_activation_buffers()
        # sample target videos for domain alignment
        target_videos,_ = target_sampler.get_sample()

        # forward pass
        preds, src_features = model(src_videos.cuda())
        _, target_features = model(target_videos.cuda())

        # Optimize for source domain accuracy
        out = F.log_softmax(preds, dim=1)
        loss_acc = F.nll_loss(out, label.cuda())
        
        # Optimize for domain alignment
        loss_mmd = mmd_loss(src_features, target_features)

        # Overall loss is calculated as: (1-mmd_weighting_factor)*classification_loss + mmd_weighting_factor*mmd_loss
        loss = (1-mmd_weighting_factor)*loss_acc + mmd_weighting_factor*loss_mmd

        loss.backward()
        optimz.step()

        
        
        if i % 50 == 0:
            print('[' +  '{:5}'.format(i * len(src_videos)) + '/' + '{:5}'.format(samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(src_data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()) +
                  ' Loss_acc ' + '{:6.4f}'.format(loss_acc.item()) +
                  ' Loss_mmd ' + '{:6.4f}'.format(loss_mmd.item())
                  )
            loss_history.append(loss.item())

def train_test_run(model, dataloaders_src, dataloaders_target, config, N_EPOCHS=5, lr=0.000005):
    start_time = time.time()

    training_losses = []

    losses_src = {
        "latest_accuracy_val": 0,
        "latest_accuracy_test": 0,
        
    }

    losses_target = {
        "latest_accuracy_val": 0,
        "latest_accuracy_test": 0,
        "best_acc_val": 0,
    }

    optimz = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(0, N_EPOCHS):
        print('Epoch:', epoch)
        optimz = exponential_decay(optimz, epoch)

        print('Started Training -------------------------------')
        train_iter(model, optimz, dataloaders_src["train"], dataloaders_target["train_sampler"], config.mmd_weighting_factor, training_losses)

        print('Started Validation -----------------------------')
        losses_src["latest_accuracy_val"] = evaluate(model, dataloaders_src["val"])
        losses_target["latest_accuracy_val"] = evaluate(model, dataloaders_target["val"])
        losses_target["best_acc_val"] = save_model_weights(model, losses_target["latest_accuracy_val"], losses_target["best_acc_val"], config.checkpoint_path, config.checkpoint_name)

        print('Started Testing --------------------------------')
        losses_src["latest_accuracy_test"] = evaluate(model, dataloaders_src["test"])
        losses_src["latest_accuracy_test"] = evaluate(model, dataloaders_target["test"])
    
    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')

    return {
        'losses_src': losses_src,
        'losses_target': losses_target,
    }


if __name__ == "__main__":
    from movinets import MoViNet
    from movinets.config import _C
    import configs.config_mmd as config

    N_EPOCHS = 5
    model = MoViNet(_C.MODEL.MoViNetA3, causal = False, pretrained = True )

    dataloaders_src = {
        "train": config.train_loader_hockey,
        "val": config.valid_loader_hockey,
        "test": config.test_loader_hockey,
    }

    dataloaders_target = {
        "train_sampler": config.train_sampler_hockey,
        "val": config.valid_loader_ucf,
        "test": config.test_loader_ucf,
    }

    train_test_run(model, dataloaders_src, dataloaders_target, config, N_EPOCHS=N_EPOCHS, lr=0.00005)
