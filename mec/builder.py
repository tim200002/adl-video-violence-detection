import torch
import torch.nn as nn
import copy


class MEC(nn.Module):
    """
    Build a MEC model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(MEC, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        # self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)
        self.encoder = base_encoder

        # build a 3-layer projector
        # prev_dim = self.encoder.classifier[-1].weight.shape[1]
        # self.encoder.classifier[-1] = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
        #                                 nn.BatchNorm1d(prev_dim),
        #                                 nn.ReLU(inplace=True), # first layer
        #                                 nn.Linear(prev_dim, prev_dim, bias=False),
        #                                 nn.BatchNorm1d(prev_dim),
        #                                 nn.ReLU(inplace=True), # second layer
        #                                 self.encoder.fc,
        #                                 nn.BatchNorm1d(dim, affine=False)) # output layer
        # self.encoder.classifier[-1][6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer


        self.teacher = copy.deepcopy(self.encoder)
        for p in self.teacher.parameters():
            p.requires_grad = False

    def clean_activation_buffers(self):
        self.encoder.clean_activation_buffers()
        self.teacher.clean_activation_buffers()        

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
        """
        _, features1 = self.encoder(x1)
        _, features2 = self.encoder(x2)
        features1 = features1.squeeze()
        features2 = features2.squeeze()
        z1 = self.predictor(features1)
        z2 = self.predictor(features2)

        with torch.no_grad():
            _, p1 = self.teacher(x1)
            _, p2 = self.teacher(x2)
            p1 = p1.squeeze()
            p2 = p2.squeeze()
        
        return z1, z2, p1.detach(), p2.detach(), features1, features2
