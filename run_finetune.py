
from movinets import MoViNet
from movinets.config import _C
from run import train_test_run

import config

if __name__ == "__main__":
    N_EPOCHS = 5
    model = MoViNet(_C.MODEL.MoViNetA3, causal = False, pretrained = True )

    # freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze the conv1, blocks
    for param in model.classifier.parameters():
        param.requires_grad = True

    train_test_run(model, config.train_loader, config.valid_loader, config.test_loader, config, N_EPOCHS=N_EPOCHS, lr=0.00005)