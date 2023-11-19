import torch
from torch import nn
import torch.nn.functional as F

from .vit import interpolate_pos_embed
from .blip import create_vit

def load_checkpoint(model,url_or_filename, prepend=''):
    '''
    This function is largly repurposed from the BLIP source,
    but only loads specific portion of the weights (text/visual encoder).

    Note that the portions on loading the momentum model is removed.
    '''
    import os

    if os.path.isfile(url_or_filename):        
        checkpoint = torch.load(url_or_filename, map_location='cpu') 
    else:
        raise RuntimeError('checkpoint url or path is invalid')
        
    state_dict = checkpoint['model']

    if prepend == 'visual_encoder.': # not applicable for vision_proj.
        state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model) #.visual_encoder) 

    for key in model.state_dict().keys():
        if prepend+key in state_dict.keys():
            if state_dict[prepend+key].shape!=model.state_dict()[key].shape:
                del state_dict[prepend+key] # delete dimension mismatched keys
                print(f"deleting {prepend+key}")
    
    for key in model.state_dict().keys():
        if not prepend+key in state_dict.keys():
            print(f"Something went wrong when loading state_dict for key {key}. Entering debug mode...")
            import pdb; pdb.set_trace()

    state_dict = {k.replace(prepend,''):v for k,v in state_dict.items()}
    msg = model.load_state_dict(state_dict,strict=False)
    print(f'load checkpoint from {url_or_filename} for {prepend}')  
    return model,msg

class BLIPImgEncoder(torch.nn.Module):
    def __init__(self, pretrained_weight_path):
        """
        This is a simple instance of the BLIP image encoder.
        Most of the configs are hardcoded for simplicity following `retrieval_coco.yaml`.

        Args:
            pretrained_weight_path (str): path for the BLIP pretrained weights
        """               
        super().__init__()

        vit = 'base'
        image_size = 384
        vit_grad_ckpt = True # do not use False
        vit_ckpt_layer = 4 # do not use 0 
        # init_lr = 1e-5 # not used unless you wish to unfreeze BLIP image encoder

        visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.visual_encoder, msg = load_checkpoint(visual_encoder, pretrained_weight_path, 'visual_encoder.')

        vision_proj = nn.Linear(vision_width, 256) # to load pretrain, embed dimension should be hardcoded
        self.vision_proj, msg = load_checkpoint(vision_proj, pretrained_weight_path, 'vision_proj.')

    def forward(self, image):
        image_embeds = self.visual_encoder(image) 
        image_feat = self.vision_proj(image_embeds[:,0,:])

        return image_feat
