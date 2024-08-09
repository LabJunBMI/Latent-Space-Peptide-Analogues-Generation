from time import time
import pandas as pd
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from transformers import AdamW
from transformers import EsmModel, EsmTokenizer
from transformers.models.esm.modeling_esm import EsmEncoder, EsmConfig

gc.enable()
# ID of GPU to use
GPU_ID = 0
device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')
# The number of sequences to generate
GEN_NUM = 3
# Model to use
ESM_MODE = "facebook/esm2_t30_150M_UR50D"
DECODER_MODEL = ""
SEQUENCES_PATH = "./data/biolip_sequences.pkl"
# Define the model with encoder and decoder parts
class EsmEncoderDecoderModel(pl.LightningModule):
    def __init__(self, tokenizer, encoder:EsmModel, learning_rate=5e-5):
        super(EsmEncoderDecoderModel, self).__init__()
        self.tokenizer = tokenizer
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        # Create decoder using the provided configuration
        decoder_config = encoder.config.to_dict()
        decoder_config["num_hidden_layers"] = 2
        # decoder_config["intermediate_size"] = 640
        self.decoder = EsmEncoder(EsmConfig(**decoder_config))
        # Add a linear layer to match the decoder output shape to the encoder input shape
        self.output_to_vocab = nn.Linear(decoder_config["hidden_size"], decoder_config["vocab_size"])
        self.learning_rate = learning_rate

    def forward(self, input_ids, attention_mask=None):
        # Pass the input through the encoder
        encoder_outputs = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        # Pass the encoder outputs through the decoder
        extended_attention_mask = self.encoder.get_extended_attention_mask(attention_mask, input_ids.size())
        decoder_outputs = self.decoder(encoder_outputs, attention_mask=extended_attention_mask).last_hidden_state
        # Transform the decoder outputs to match the input shape of the tokenizer
        transformed_outputs = self.output_to_vocab(decoder_outputs)
        return transformed_outputs

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask = batch
        # Forward pass
        outputs = self(input_ids, attention_mask)
        # Calculate loss (example using CrossEntropyLoss)
        print(outputs.view(-1, outputs.size(-1)), outputs.view(-1, outputs.size(-1)).shape)
        print(input_ids.view(-1), input_ids.view(-1).shape)
        loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), input_ids.view(-1), ignore_index=self.tokenizer.pad_token_id)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask = batch
        # Forward pass
        outputs = self(input_ids, attention_mask=attention_mask)
        # Calculate loss (example using CrossEntropyLoss)
        loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), input_ids.view(-1), ignore_index=self.tokenizer.pad_token_id)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
    
def has_repeated_AA(s: str, threshold: float=0.5) -> bool:
    # Calculate the threshold count
    threshold_count = len(s) * threshold
    # Create a dictionary to count occurrences of each character
    char_count = {}
    # Count occurrences of each character
    for char in s:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1
    # Check if any character exceeds the threshold count
    for count in char_count.values():
        if count > threshold_count:
            return True
    return False

def has_consecutive_AA(s: str, threshold: float=0.3) -> bool:
    # Calculate the threshold count
    threshold_count = len(s) * threshold
    # Initialize variables to track the current character and its consecutive count
    max_consecutive_count = 0
    current_char = ''
    current_consecutive_count = 0
    # Iterate through the string to count consecutive characters
    for char in s:
        if char == current_char:
            current_consecutive_count += 1
        else:
            current_char = char
            current_consecutive_count = 1
        # Update the max consecutive count
        if current_consecutive_count > max_consecutive_count:
            max_consecutive_count = current_consecutive_count
    # Check if the max consecutive count exceeds the threshold count
    return max_consecutive_count > threshold_count

def get_model_tokenizer(encoder_model_name):
    # Load the pre-trained ESM model
    encoder_model = EsmModel.from_pretrained(encoder_model_name)
    # Load the tokenizer
    tokenizer = EsmTokenizer.from_pretrained(encoder_model_name)
    model = EsmEncoderDecoderModel(tokenizer, encoder_model)
    return model, tokenizer

def get_embedding(seq):
    token = tokenizer(
        seq, return_tensors="pt", 
        max_length=256, padding="max_length", truncation=True
    )
    extended_attention_mask = model.encoder.get_extended_attention_mask(
        token["attention_mask"].to(device), 
        token["input_ids"].size()
    )
    embedding = model.encoder(
        token["input_ids"].to(device), 
        attention_mask=token["attention_mask"].to(device)
    ).last_hidden_state
    return embedding, extended_attention_mask

def generate_seq(
    input_seq, embedding, extended_attention_mask, gen_num=3,
    noise_start=0.5, noise_step=0.1, noise_timestep=2000, time_limit=10000
):
    total_start = time()
    total_step=0
    # Load the model and tokenizer
    noise_add = 0
    i=0
    res=[]
    step=0
    
    while i<gen_num:
        # Add noise to the hidden states
        noise = torch.randn_like(embedding)
        noise = noise - noise.min()  # Shift noise to be non-negative
        noise = noise / noise.max()  # Normalize noise to [0, 1]
        noise = noise * 2 - 1  # Shift noise to [-1, 1]
        noise += torch.rand(1).item()*2-1 # Add a random shift to the noise
        noise = noise * (noise_start+(noise_add*noise_step))  # Adjust the scale of noise as needed
        noised_embedding = embedding + noise
        noised_embedding = noised_embedding.to(device)
        # Prepare the encoder outputs with the noised hidden states
        decoder_outputs = model.decoder(
            noised_embedding, 
            attention_mask=extended_attention_mask
        ).last_hidden_state
        # transform the hidden state to vocab
        transformed_outputs = model.output_to_vocab(decoder_outputs)
        pred_ids = torch.functional.F.softmax(transformed_outputs, 2)[0].argmax(axis=1)
        output_seq = "".join([tokenizer.all_tokens[i] for i in pred_ids][1:len(input_seq)+1])
        is_input_weird = has_consecutive_AA(input_seq) or (has_repeated_AA(input_seq))
        is_weird = has_consecutive_AA(output_seq) or (has_repeated_AA(output_seq))
        is_weird = is_weird and (not is_input_weird)# if input is weird, the resuld could be weird
        if (output_seq != input_seq) and (output_seq not in res) and (not is_weird):
            print("Noised and Reconstructed Output:", output_seq)
            res.append(output_seq)
            i+=1
            step=0
        if(step>noise_timestep):
            noise_add+=1
            step = 0
            print(f"Increasing Noise to: {noise_start+(noise_add*noise_step)}")
        if(step>time_limit):
            print("Reach time limit")
            break
        step+=1
        total_step+=1
        del noised_embedding
    print("Original Input:", input_seq)
    print("Time cost:", time()-total_start)
    print("Total step:", total_step)
    return res

if __name__ == "__main__":
    model, tokenizer = get_model_tokenizer(ESM_MODE)
    model = model.to(device)
    model.load_state_dict(torch.load(DECODER_MODEL))
    model = model.eval()
    df = pd.read_pickle(SEQUENCES_PATH)
    res = []
    for idx, seq in enumerate(df.ligand_sequence):
        print(f"\nGenerating {idx}, seqeucne:{seq}")
        embedding,mask = get_embedding(seq)
        res.append(generate_seq(seq, embedding, mask, gen_num=GEN_NUM))

    df["generated_seq"] = res
    df.to_pickle(f"esm_generated_{GEN_NUM}.pkl")