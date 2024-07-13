import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertForSequenceClassification, AdamW
from tqdm import trange
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from sklearn.metrics import matthews_corrcoef




# Load the pre-trained BERT model for sequence classification with two labels
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Get all of the model's parameters as a list of tuples.
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']

# Create the optimizer parameters with weight decay for all parameters except bias, gamma, and beta
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

# Initialize the AdamW optimizer
optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, correct_bias=False)

# Print the optimizer to verify
print(optimizer)


file_path = "C:/Users/jorda/OneDrive/Desktop/CodeTech/NATURAL LANGUAGE PROCESSING/BERT/Coca cola (csv)/in_domain_train.tsv"
df = pd.read_csv(file_path, delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
print(df.head())
print(df.shape)
print(df.sample(10))

# Create lists for sentences and labels
sentences = df['sentence'].values
labels = df['label'].values
# Adding special tokens at the beginning and end of each sentence for BERT
sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
print(sentences[:5])
print(labels[:5])

# Loading the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
# Tokenize the sentences
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
print("Tokenize the first sentence:")
print(tokenized_texts[0])
MAX_LEN = 128
# Tokenize the sentences
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
# Convert the tokens to input IDs
input_ids = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokenized_texts]
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
# Print the first padded input ID sequence to verify
print("Padded and truncated input IDs for the first sentence:")
print(input_ids[0])

# Create attention masks
attention_masks = []
# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
    seq_mask = [float(i > 0) for i in seq]
    attention_masks.append(seq_mask)
# Print the first attention mask to verify
print("Attention mask for the first sentence:")
print(attention_masks[0])

# Useing train_test_split to split the data into training and validation sets
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(
    input_ids, labels, random_state=2018, test_size=0.1
)

train_masks, validation_masks, _, _ = train_test_split(
    attention_masks, input_ids, random_state=2018, test_size=0.1
)
# Print the shapes to verify the split
print("Training input shape:", train_inputs.shape)
print("Validation input shape:", validation_inputs.shape)
print("Training labels shape:", train_labels.shape)
print("Validation labels shape:", validation_labels.shape)
print("Training masks shape:", len(train_masks))
print("Validation masks shape:", len(validation_masks))

# Convert all of our data into torch tensors, the required datatype for our model
train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

# Print the types of the converted tensors to verify
print("Training inputs tensor type:", train_inputs.type())
print("Validation inputs tensor type:", validation_inputs.type())
print("Training labels tensor type:", train_labels.type())
print("Validation labels tensor type:", validation_labels.type())
print("Training masks tensor type:", train_masks.type())
print("Validation masks tensor type:", validation_masks.type())

# Select a batch size for training (16or32) 
batch_size = 32
# Create the DataLoader for our training set
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
# Create the DataLoader for our validation set
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
print("Number of batches in training dataloader:", len(train_dataloader))
print("Number of batches in validation dataloader:", len(validation_dataloader))

# Load the pre-trained BERT model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
# Obtain all of the model's parameters as a list of tuples.
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
# Create the optimizer 
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
# Initialize the AdamW optimizer
optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, correct_bias=False)
print(optimizer)

# calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

seed_val = 42
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
####
# Load data
df = pd.read_csv("C:/Users/jorda/OneDrive/Desktop/CodeTech/NATURAL LANGUAGE PROCESSING/BERT/Coca cola (csv)/in_domain_train.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])

# Create sentence and label lists
sentences = df.sentence.values
sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
labels = df.label.values

# Initialize tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)

# Tokenize sentences
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

# Pad tokenized texts to ensure equal length
MAX_LEN = 128  # Example max length
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# Create attention masks
attention_masks = []
for seq in input_ids:
    seq_mask = [float(i > 0) for i in seq]
    attention_masks.append(seq_mask)

# Split data into training and validation sets
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=2018, test_size=0.1)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=2018, test_size=0.1)

# Convert data to torch tensors
train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

# Create DataLoader for training and validation sets
batch_size = 32
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

# Load model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 2  # Define number of epochs
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Initialize variables
train_loss_set = []

# Training loop
for epoch in trange(epochs, desc="Epoch"):
    model.train()
    tr_loss = 0
    nb_tr_steps = 0

    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        train_loss_set.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        tr_loss += loss.item()
        nb_tr_steps += 1

    print(f"Train loss: {tr_loss / nb_tr_steps}")

    # Validation
    model.eval()
    eval_accuracy = 0
    nb_eval_steps = 0

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask)
            logits = outputs.logits

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print(f"Validation Accuracy: {eval_accuracy / nb_eval_steps}")
    
plt.figure(figsize=(15, 8))
plt.title("Training loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.plot(train_loss_set)
plt.show()

df = pd.read_csv("C:/Users/jorda/OneDrive/Desktop/CodeTech/NATURAL LANGUAGE PROCESSING/BERT/Coca cola (csv)/out_of_domain_dev.tsv", delimiter='\t', 
                 header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
# Sentence and label lists
sentences = df.sentence.values
labels = df.label.values

# Adding special tokens at the beginning and end of each sentence for BERT
sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]

# Loading the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Tokenize the sentences
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

MAX_LEN = 128

# Convert the tokens to their index numbers in the BERT vocabulary
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

# Pad our input tokens
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# Attention masks created
attention_masks = []

# Creating a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
    seq_mask = [float(i > 0) for i in seq]
    attention_masks.append(seq_mask)

# Convert to torch tensors
prediction_inputs = torch.tensor(input_ids)
prediction_masks = torch.tensor(attention_masks)
prediction_labels = torch.tensor(labels)

# Define batch size
batch_size = 32  

# Create DataLoader
prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

#Prediction on test set

# Put model in evaluation mode
model.eval()

# Initialize lists to hold predictions and true labels
predictions, true_labels = [], []

# Predict
for batch in prediction_dataloader:
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
    
    # Telling the model not to compute or store gradients, saving memory and speeding up prediction
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask)
        logits = outputs.logits

    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    
    # Store predictions and true labels
    predictions.append(logits)
    true_labels.append(label_ids)

# Flatten predictions and true labels
predictions = np.concatenate(predictions, axis=0)
true_labels = np.concatenate(true_labels, axis=0)

# Convert logits to predicted labels
predicted_labels = np.argmax(predictions, axis=1)

# Compute Matthews correlation coefficient
matthews = matthews_corrcoef(true_labels, predicted_labels)

print(f"Matthews correlation coefficient: {matthews:.4f}")