import torch
from torch import nn
import torch.nn.functional as F

from .med import init_tokenizer
from .med import BertConfig, BertModel

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

    for key in model.state_dict().keys():
        if prepend+key in state_dict.keys():
            if state_dict[prepend+key].shape!=model.state_dict()[key].shape:
                del state_dict[prepend+key]
                print(f"deleting {prepend+key}")

    for key in model.state_dict().keys():
        if not prepend+key in state_dict.keys():
            print(f"Something went wrong when loading state_dict for key {key}. Entering debug mode...")
            import pdb; pdb.set_trace()

    state_dict = {k.replace(prepend,''):v for k,v in state_dict.items()}
    msg = model.load_state_dict(state_dict,strict=False)
    print(f'load checkpoint from {url_or_filename} for {prepend}')
    return model,msg

class BLIPTextEncoder(torch.nn.Module):
    def __init__(self, pretrained_weight_path, med_config_path, use_pretrained_proj_layer=True):
        """
        This is a simple instance of the BLIP text encoder.
        Most of the configs are inherited from `med_config.json`.

        Args:
            pretrained_weight_path (str): path for the BLIP pretrained weights
            med_config_path (str): path for the mixture of encoder-decoder model's configuration file
            use_pretrained_proj_layer (bool): whether to use the pretrained projection layer
        """               
        super().__init__()

        self.tokenizer = init_tokenizer()
        med_config = med_config_path
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = 768 # vision_width
        text_encoder = BertModel(config=encoder_config, add_pooling_layer=False) 
        self.text_encoder, msg = load_checkpoint(text_encoder, pretrained_weight_path, 'text_encoder.')

        if use_pretrained_proj_layer:
            text_proj = nn.Linear(768, 256) # (text_width, embed_dim)
            self.text_proj, msg = load_checkpoint(text_proj, pretrained_weight_path, 'text_proj.')
        else:
            self.text_proj = nn.Linear(768, 640) # assuming using clip img encoder, so get 640 out here

    def forward(self, caption, max_length, device):
        caption = self.tokenizer(caption, padding='longest', truncation=True, max_length=max_length, 
                                return_tensors="pt").to(device) 
        '''
        Default: use [CLS]
        Uncomment the next line to use the BLIP-dedicated [ENCODE] special token.
        In our testings, we observe no noticable performance difference between [CLS] and [ENCODE] during finetuning.
        '''
        # caption.input_ids[:,0] = self.tokenizer.enc_token_id 

        caption_embedding = self.text_encoder(caption.input_ids, attention_mask = caption.attention_mask, return_dict = True, mode = 'text')
        caption_embedding_pooled = self.text_proj(caption_embedding.last_hidden_state[:,0,:])
        
        return caption_embedding_pooled