from transformers import T5Tokenizer, T5ForConditionalGeneration
from time import time
import pandas as pd
import torch
import gc
import re

gc.enable()

# ID of GPU to use
GPU_ID = 0
device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')
# The number of sequences to generate
GEN_NUM = 3
# Model to use
MODEL_PATH = ""
SEQUENCES_PATH = "./data/biolip_sequences.pkl"

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

def get_embedding(seq:str):
    #"C N C K R F P Q C P L N F L C"
    # Define your input
    sequences_Example = [" ".join(seq)]
    sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]
    input_seq = sequences_Example[0]

    # Tokenize the input text
    tokens = tokenizer(input_seq, add_special_tokens=True, padding=True, return_tensors="pt")
    tokens = tokens.to(device)

    # Pass the input through the encoder
    encoder_outputs = model.encoder(
        input_ids=tokens.input_ids,
        attention_mask=tokens.attention_mask
    )
    # Extract the hidden states
    encoder_hidden_states = encoder_outputs.last_hidden_state
    del tokens, encoder_outputs
    return encoder_hidden_states

def generate_seq(
    input_seq, embedding, gen_num=3, 
    noise_start=0.5, noise_step=0.1, noise_timestep=50, time_limit=500
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
        noised_encoder_outputs = (noised_embedding,)  # Encoder outputs are usually a tuple
        # Prepare the decoder input (usually starts with <pad> token)
        decoder_input_ids = tokenizer("<pad>", return_tensors='pt').input_ids
        decoder_input_ids = decoder_input_ids.to(device)
        # Generate the output sequence using the noised hidden states from the encoder
        output_ids = model.generate(
            input_ids=decoder_input_ids, 
            encoder_outputs=noised_encoder_outputs,
            max_length=len(input_seq)+1
        )
        # Decode the output ids to get the text
        output_seq:str = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        output_seq = output_seq.replace(" ", "")
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
        del decoder_input_ids, noised_embedding
    print("Original Input:", input_seq)
    print("Time cost:", time()-total_start)
    print("Total step:", total_step)
    return res


if __name__ == "__main__":
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, local_files_only=True)
    model = model.eval()
    model = model.to(device)
    df = pd.read_pickle(SEQUENCES_PATH)
    res = []
    for idx, seq in enumerate(df.ligand_sequence):
        print(f"\nGenerating {idx}, seqeucne:{seq}")
        res.append(generate_seq(seq, get_embedding(seq), gen_num=GEN_NUM))

    df["generated_seq"] = res
    df.to_pickle(f"protT5_generated_{GEN_NUM}.pkl")