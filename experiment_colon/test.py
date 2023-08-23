import re
from datetime import datetime

from sentence_transformers import models, SentenceTransformer, losses
from sentence_transformers.datasets import DenoisingAutoEncoderDataset
from torch.utils.data import DataLoader

import datasets
import nltk

nltk.download('punkt')

train_batch_size = 16
num_epochs = 1
model_name = 'bert-base-uncased'
model_save_path = 'output/training_continue_training-' + model_name + '-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

oscar = datasets.load_dataset(
    'oscar',
    'unshuffled_deduplicated_en',
    split='train',
    streaming=True
)

splitter = re.compile(r'\.\s?\n?')

# Create dataset with size 10_000
num_sentences = 0
sentences = []
for row in oscar:
    new_sentences = splitter.split(row['text'])
    new_sentences = [line for line in new_sentences if len(line) > 10]
    sentences.extend(new_sentences)
    num_sentences += len(new_sentences)
    if num_sentences > 100_000:
        # Sentence transformers recommends 10-100K sentences for training
        break

# print(sentences[10:])

# Create a DenoisingAutoEncoderDataset with 0.6 prob to delete words
train_data = DenoisingAutoEncoderDataset(sentences)

loader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size, drop_last=True)

# Build sentence Transformer
bert = models.Transformer(model_name, max_seq_length=256)
pooling_model = models.Pooling(bert.get_word_embedding_dimension(), 'cls')
model = SentenceTransformer(modules=[bert, pooling_model])

loss = losses.DenoisingAutoEncoderLoss(model, tie_encoder_decoder=True)

# Train the model
model.fit(
    train_objectives=[(loader, loss)],
    epochs=num_epochs,
    weight_decay=0,
    scheduler='constantlr',
    optimizer_params={'lr': 3e-05},
    save_best_model=True,
    show_progress_bar=True,
    output_path=model_save_path
)

model.save(model_save_path)
loss.decoder.save_pretrained(f"{model_save_path}_decoder")
