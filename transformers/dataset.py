import torch 
import torch.nn as nn
from torch.utils.data import Dataset 

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        self.sos_token = torch.tensor(tokenizer_src.token_to_id('[SOS]'), dtype=torch.int64)
        self.eos_token = torch.tensor(tokenizer_src.token_to_id('[EOS]'), dtype=torch.int64)
        self.pad_token = torch.tensor(tokenizer_src.token_to_id('[PAD]'), dtype=torch.int64)

    def __len__(self):
        return len(self.ds) 
    
    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        enc_num_padding_tokens = max(0,self.seq_len - len(enc_input_tokens) - 2)
        dec_num_padding_tokens = max(0,self.seq_len - len(dec_input_tokens) - 1)
        
        # if not enc_num_padding_tokens >= 0:
        #     breakpoint()
        #     raise ValueError(f"enc_num_padding_tokens should be >= 0, got {enc_num_padding_tokens}")
        
        encoder_input = torch.cat([
            self.sos_token.unsqueeze(0), # Start of sentence token
            torch.tensor(enc_input_tokens, dtype=torch.int64), # Encoder input tokens
            self.eos_token.unsqueeze(0), # End of sentence token
            torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64) # Padding tokens
        ])

        decoder_input = torch.cat([
            self.sos_token.unsqueeze(0), # Start of sentence token
            torch.tensor(dec_input_tokens, dtype=torch.int64), # Decoder input tokens
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64) # Padding tokens
        ])

        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64), # Decoder input tokens
            self.eos_token.unsqueeze(0), # End of sentence token
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64) # Padding tokens
        ])
        
        
        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size()[0]),
            'label': label,
            'src_text': src_text,
            'tgt_text': tgt_text
        }

def causal_mask(seq_len):
    mask = torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1).type(torch.int)
    return mask == 0
    