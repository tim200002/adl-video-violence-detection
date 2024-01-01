
import logging
import torch
from models.movinets.config import _C
from models.movinets.models import MoViNet


def get_model(model_name="A1", checkpoint_path=None, causal=False):
    if model_name == "A1":
        logging.info("Loading MoViNet A1")
        model = MoViNet(_C.MODEL.MoViNetA1, causal = causal, pretrained = True)
    elif model_name == "A3":
        logging.info("Loading MoViNet A3")
        model = MoViNet(_C.MODEL.MoViNetA3, causal = causal, pretrained = True)
    else:
        raise ValueError("Model name not recognized.")
    
    # change prediction head for 2 class prediction
    model.classifier[3] = torch.nn.Conv3d(2048, 1, (1,1,1))
    
    model.cuda()
    if checkpoint_path is not None:
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
    
    return model