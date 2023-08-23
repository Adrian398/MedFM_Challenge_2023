import logging
import os
import random

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoConfig, AutoModelForCausalLM

logger = logging.getLogger(__name__)

model_save_path = 'experiment_colon/refine-net/output/training_continue_training-bert-base-uncased-2023-08-23_14-17-40'
decoder_suffix = 'decoder'
model = SentenceTransformer(model_save_path)

sentences = ['This framework generates embeddings for each input sentence',
             'Sentences are passed as a list of string.',
             'The quick brown fox jumps over the lazy dog.']


def create_noisy_sentence(sentence, drop_probability=0.2):
    words = sentence.split()
    noisy_words = [word for word in words if random.random() > drop_probability]
    return ' '.join(noisy_words)


# Create noisy versions of the original sentences
noisy_sentences = [create_noisy_sentence(sentence) for sentence in sentences]

# Initialize Decoder
decoder_path = os.path.join(model_save_path + '_' + decoder_suffix)
decoder_config = AutoConfig.from_pretrained(decoder_path)
decoder_config.is_decoder = True
decoder_config.add_cross_attention = True
kwargs_decoder = {'config': decoder_config}
try:
    decoder = AutoModelForCausalLM.from_pretrained(decoder_path, **kwargs_decoder)
except ValueError as e:
    logger.error(
        f'Model name or path "{model_save_path}" does not support being as a decoder. Please make sure the decoder '
        f'model has an "XXXLMHead" class.')
    raise e

# Tokenize noisy sentences
sentence_features = model.tokenize(noisy_sentences)

# Encode the noisy sentences
reps = model(sentence_features)['sentence_embedding']

input_ids = torch.full((reps.size(0), 1), model.tokenizer.cls_token_id, dtype=torch.long)

target_length = 12
for _ in range(target_length):
    decoder_outputs = decoder(
        input_ids=input_ids,
        encoder_hidden_states=reps[:, None],
        use_cache=True
    )
    next_token_logits = decoder_outputs.logits[:, -1, :]
    next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
    input_ids = torch.cat([input_ids, next_tokens], dim=-1)

# Decode the input IDs to sentences
reconstructed_sentences = []
for ids in input_ids:
    reconstructed_sentences.append(model.tokenizer.decode(ids, skip_special_tokens=True))

for original, noisy, reconstructed in zip(sentences, noisy_sentences, reconstructed_sentences):
    print(f"Original: {original}")
    print(f"Noisy: {noisy}")
    print(f"Reconstructed: {reconstructed}")
    print("-" * 80)
