import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

from transformers import AdamW
from transformers import EsmModel, EsmTokenizer
from transformers.models.esm.modeling_esm import EsmEncoder, EsmConfig

import numpy as np
import gc

# Fasta file path, UniprotKB/Swiss-Prot: https://www.uniprot.org/help/downloads
UNIPROTKB_PATH = "./uniprot_sprot.fasta"
# Model save path
MODEL_PATH = ""
# ID of GPU to use
GPU_ID = 7

class FastaDataset(Dataset):
    def __init__(self, fasta_file, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sequences = self._load_fasta(fasta_file)
    def _load_fasta(self, fasta_file):
        sequences = []
        with open(fasta_file, 'r') as f:
            seq = ''
            for line in f:
                if line.startswith('>'):
                    if seq:
                        sequences.append(seq)
                        seq = ''
                else:
                    seq += line.strip()
            if seq:
                sequences.append(seq)
        return sequences
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        inputs = self.tokenizer(sequence, return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True)
        return inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0)

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

def get_model_tokenizer(encoder_model_name):
    # Load the pre-trained ESM model
    encoder_model = EsmModel.from_pretrained(encoder_model_name)
    # Load the tokenizer
    tokenizer = EsmTokenizer.from_pretrained(encoder_model_name)
    model = EsmEncoderDecoderModel(tokenizer, encoder_model)
    return model, tokenizer

def get_dataloader(file_path, tokenizer, batch_size):
    dataset = FastaDataset(file_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def train_model(model, dataloader, num_epochs):
    trainer = pl.Trainer(
        max_epochs=num_epochs, 
        accelerator="gpu",
        check_val_every_n_epoch=1,
        devices=[7],
    )
    trainer.fit(model, dataloader)
    return trainer

if __name__ =="__main__":
    model, tokenizer = get_model_tokenizer("facebook/esm2_t30_150M_UR50D")
    dataloader = get_dataloader(UNIPROTKB_PATH, tokenizer, 64)
    trainer = train_model(model, dataloader, 1)
    torch.save(model.state_dict(), MODEL_PATH)
