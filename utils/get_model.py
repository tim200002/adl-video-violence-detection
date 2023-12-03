
import torch
from movinets.config import _C
from movinets.models import MoViNet


def get_model(checkpoint_path=None):
    model = MoViNet(_C.MODEL.MoViNetA3, causal = False, pretrained = True )
    model.cuda()
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        #model.load_state_dict(checkpoint['model_state_dict'], map_location=torch.device('cuda')
        model.load_state_dict(checkpoint)
    return model