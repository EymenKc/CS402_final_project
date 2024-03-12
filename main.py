import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Load CaSiNo dataset
casino_dataset = load_dataset("casino")

# Get unannotated dialogues
unannotated_dialogues = casino_dataset["train"]["chat_logs"]

paired_sequences = []

# Pair up utterances from two parties
for dialogue in unannotated_dialogues:
    for i in range(1, len(dialogue)):
        if i % 2 == 0:  # Assuming the dialogues alternate between the two parties
            paired_sequences.append((dialogue[i - 1]["text"], dialogue[i]["text"]))

# Initialize RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Tokenize paired sequences
tokenized_pairs = tokenizer(paired_sequences, truncation=True, padding=True)

# Define custom PyTorch dataset
class DialogueDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_sequences):
        self.tokenized_sequences = tokenized_sequences

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.tokenized_sequences.items()}

    def __len__(self):
        return len(self.tokenized_sequences.input_ids)

dataset = DialogueDataset(tokenized_pairs)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.3
)

# Get the GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained RoBERTa model
model = RobertaForMaskedLM.from_pretrained("roberta-base")

# Move the model to the GPU
model.to(device)
print("device:", device)
print("models device:", next(model.parameters()).device)

# Load pre-trained RoBERTa model
model = RobertaForMaskedLM.from_pretrained("roberta-base").cuda()  # Move model to GPU

# Training arguments
training_args = TrainingArguments(
    output_dir="./roberta_idpt",
    overwrite_output_dir=True,
    num_train_epochs=20,  # Adjust epochs as needed
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./roberta_casino_idpt")

