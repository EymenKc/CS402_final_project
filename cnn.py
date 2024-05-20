from datasets import load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from gensim.models import Word2Vec
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import sys
from transformers import AutoModel, AutoTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
nltk.download('punkt')



class CasinoDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(64, 128, 3) # (input_channels, output_channels, kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 64, 3) # (input_channels, output_channels, kernel_size)
        self.fc1 = nn.Linear(64 * 4 * 42, 256)  # Adjusted input size for fc1
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 1) 
        

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))  
        x = x.view(-1, 64 * 4 * 42)  # Adjusted input size for fc1
        x = F.relu(self.fc1(x))   
        x = F.relu(self.fc2(x))      
        x = self.fc3(x)
        x = self.fc4(x)
        x = torch.squeeze(x, dim=1)  # Squeeze the singleton dimension
        x = torch.sigmoid(x)
        return x


class BERTReducer(nn.Module):
    def __init__(self, input_size=768, output_size=64):
        super(BERTReducer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)
    

def data_prep():
    # Load CaSiNo dataset
    casino_dataset = load_dataset("casino")

    labels = ["small-talk", "showing-empathy", "promote-coordination", "no-need", "elicit-pref", "uv-part", "vouch-fair", "self-need", "other-need", "non-strategic"]

    annotated_dialogue = []
    annotated_labels = []
            
    # Get annotated dialogues
    for dialogue in casino_dataset["train"]["annotations"]:
        if len(dialogue) == 0:
            continue
        annotated_sequences = []
        sequence_labels = []
        for sequence, label_str in dialogue:
            # Split the label string and filter out labels not in the 'labels' set
            current_labels = [label.strip() for label in label_str.split(",") if label.strip() in labels]
            
            # Convert labels to binary encoding            
            annotated_sequences.append(sequence)
            # remove empty strings that are not in the labels
            current_labels = [label for label in current_labels if label in labels]
            label_list = [0] * len(labels)
            for label in current_labels:
                label_list[labels.index(label)] = 1
            sequence_labels.append(label_list)

        annotated_dialogue.append(annotated_sequences)
        annotated_labels.append(sequence_labels)

    return annotated_dialogue, annotated_labels


def w2v_embeddings(annotated_dialogues, vector_size=64):
    # Tokenize the sentences
    annotated_sentences = []
    for dialogue in annotated_dialogues:
        for sentence in dialogue:
            annotated_sentences.append(word_tokenize(sentence))

    # Train word2vec model
    word2vec_model = Word2Vec(annotated_sentences, vector_size=vector_size, window=5, min_count=1, workers=4) 

    # Get word vectors
    word_vectors = word2vec_model.wv

    return word_vectors


def word_embedding(annotated_dialogues, vector_size, embedding_model):
    if embedding_model == "word2vec":
        word_vectors = w2v_embeddings(annotated_dialogues, vector_size)
    #elif embedding_model == "bert":
    #    word_vectors = bert_embeddings(annotated_dialogues, vector_size, embedding_model)

    return word_vectors


def create_dialogue_image(annotated_dialogues, word_vectors, dialogue_max_length=24, sentence_max_length=174):
    # Create dialogue image
    #dialogue_images = []
    #for dialogue in annotated_dialogues:
    #    dialogue_image = []
    #    for sentence in dialogue:
    #        sentence_image = []
    #        for word in word_tokenize(sentence):
    #            sentence_image.append(word_vectors[word])
    #        dialogue_image.append(sentence_image)
    #    dialogue_images.append(dialogue_image)

    # Create dialogue image
    dialogue_images = []
    for dialogue in annotated_dialogues:
        for i in range(len(dialogue)):
            dialogue_image = []
            sentence_index = 0
            for sentence in dialogue:
                if i >= sentence_index:
                    sentence_image = []
                    for word in word_tokenize(sentence):
                        sentence_image.append(word_vectors[word])
                    dialogue_image.append(sentence_image)
                sentence_index += 1
            dialogue_images.append(dialogue_image)


    # Pad the dialogue image
    for dialogue_image in dialogue_images:
        for sentence in dialogue_image:
            while len(sentence) < sentence_max_length:
                sentence.append(np.zeros((64,), dtype=np.float32))  # Use np.zeros to pad instead of list comprehension
    
    for dialogue in dialogue_images:
        while len(dialogue) < dialogue_max_length:
            dialogue.append([np.zeros((64,), dtype=np.float32)] * sentence_max_length)  # Use np.zeros to pad instead of list comprehension

    # Convert the padded dialogue images to a NumPy array
    dialogue_images_numpy = np.array(dialogue_images)

    # Convert the NumPy array into a PyTorch tensor
    dialogue_images_tensor = torch.from_numpy(dialogue_images_numpy)

    # Reshape the tensor to match the desired shape
    dialogue_images_tensor = dialogue_images_tensor.permute(0, 3, 1, 2)

    return dialogue_images_tensor


def bert_create_dialogue_image(annotated_dialogues, dialogue_max_length=24, sentence_max_length=768):

    model_checkpoint = "bert-base-uncased"
    tokenizer_checkpoint = "bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint, do_lower_case=False)  # tokenizer for protein sequences
    model = AutoModel.from_pretrained(model_checkpoint)  # model for protein sequences

    # Move the model to the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    # Create dialogue image
    dialogue_images = []
    for dialogue in annotated_dialogues:
        dialogue_image = []
        for sentence in dialogue:
            # Tokenize the sequence
            tokenized_sequence = tokenizer(sentence, add_special_tokens=False, return_tensors="pt", padding="max_length", truncation=True, max_length=64)

            # Encode the sequence
            outputs = model(**tokenized_sequence.to(device))


            # Get the pooler output
            last_hidden_state = outputs.last_hidden_state.detach().cpu().numpy()

            dialogue_image.append(last_hidden_state)

            # Free up CUDA memory
            del tokenized_sequence
            del outputs
            del last_hidden_state
            torch.cuda.empty_cache()
            
        dialogue_images.append(dialogue_image)

    # Convert the padded dialogue images to a NumPy array
    dialogue_images_numpy = np.array(dialogue_images)

    # Convert the NumPy array into a PyTorch tensor
    dialogue_images_tensor = torch.from_numpy(dialogue_images_numpy)

    # Reshape the tensor to match the desired shape
    dialogue_images_tensor = dialogue_images_tensor.permute(0, 3, 1, 2)

    return dialogue_images_tensor


def create_label_tensor(annotated_labels, chosen_label="small-talk", dialogue_max_length=24):
    labels = ["small-talk", "showing-empathy", "promote-coordination", "no-need", "elicit-pref", "uv-part", "vouch-fair", "self-need", "other-need", "non-strategic"]
    
    label_index = labels.index(chosen_label)

    all_labels = []
    
    for dialogue in annotated_labels:
        for i in range(len(dialogue)):
            sentence_index = 0
            for sequence in dialogue:
                if i == sentence_index:
                    dialogue_label = sequence[label_index]
                sentence_index += 1
            all_labels.append(dialogue_label)



    # Create dialogue image
    #dialogue_images = []
    #for dialogue in annotated_dialogues:
    #    for i in range(len(dialogue)):
    #        dialogue_image = []
    #        sentence_index = 0
    #        for sentence in dialogue:
    #            if i >= sentence_index:
    #                sentence_image = []
    #                for word in word_tokenize(sentence):
    #                    sentence_image.append(word_vectors[word])
    #                dialogue_image.append(sentence_image)
    #            sentence_index += 1
    #        dialogue_images.append(dialogue_image)


    all_labels = np.array(all_labels)
    label_tensor = torch.from_numpy(all_labels)

    # Turn values into float32
    label_tensor = label_tensor.type(torch.float32)

    return label_tensor


def forward_pass(model, learning_rate, num_epochs, train_loader, device):

    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # origin shape: [4, 3, 32, 32] = 4, 3, 1024
            # input_layer: 3 input channels, 6 output channels, 5 kernel size
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

        if epoch % 50 == 0:
            print (f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')

    return model


def test_model(model, test_loader, device, threshold=0.5):
    # Test
    model.eval()
    with torch.no_grad():
        f1_scores = []
        accuracies = []
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            # Convert outputs to binary predictions using threshold
            binary_outputs = (outputs > threshold).int()

            # Flatten the labels and binary_outputs arrays
            labels_flat = labels.view(-1).cpu().numpy()
            binary_outputs_flat = binary_outputs.view(-1).cpu().numpy()

            #print("Labels:", labels_flat)
            #print("Binary Outputs:", binary_outputs_flat)
            

            # Calculate F1 score and accuracy
            f1 = f1_score(labels_flat, binary_outputs_flat, average='macro')
            accuracy = accuracy_score(labels_flat, binary_outputs_flat)

            f1_scores.append(f1)
            accuracies.append(accuracy)

        avg_f1 = sum(f1_scores) / len(f1_scores)
        avg_accuracy = sum(accuracies) / len(accuracies)
        print(f'Average F1 Score: {avg_f1}, Average Accuracy: {avg_accuracy}')


def k_fold_cross_validation(batch_size, vector_size, learning_rate, num_epochs, chosen_label, device, embedding_model, k=5):
    # Load the annotated dialogues and labels
    annotated_dialogues, annotated_labels = data_prep()
        
    # Load the word vectors
    word_vectors = word_embedding(annotated_dialogues, vector_size, embedding_model)

    dialogue_max_length = max(len(dialogue) for dialogue in annotated_dialogues)
    sentence_max_length = max(len(word_tokenize(sentence)) for dialogue in annotated_dialogues for sentence in dialogue)

    # Split the data into k folds
    fold_size = int(len(annotated_dialogues) / k)
    for i in range(k):
        print("Fold: ", i+1)
        # Split the data into training and testing sets
        train_dialogues = annotated_dialogues[:i * fold_size] + annotated_dialogues[(i + 1) * fold_size:]
        train_labels = annotated_labels[:i * fold_size] + annotated_labels[(i + 1) * fold_size:]
        test_dialogues = annotated_dialogues[i * fold_size:(i + 1) * fold_size]
        test_labels = annotated_labels[i * fold_size:(i + 1) * fold_size]

        # Create the dialogue images
        train_dialogue_images_tensor = create_dialogue_image(train_dialogues, word_vectors, dialogue_max_length, sentence_max_length)
        test_dialogue_images_tensor = create_dialogue_image(test_dialogues, word_vectors, dialogue_max_length, sentence_max_length)

        # Create the label tensor
        train_label_tensor = create_label_tensor(train_labels, chosen_label, dialogue_max_length)
        test_label_tensor = create_label_tensor(test_labels, chosen_label, dialogue_max_length)

        # Create the datasets
        train_dataset = CasinoDataset(train_dialogue_images_tensor, train_label_tensor)
        test_dataset = CasinoDataset(test_dialogue_images_tensor, test_label_tensor)

        # Create the data loaders
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

        # Initialize the model
        model = ConvNet().to(device)

        # Train the model
        model = forward_pass(model, learning_rate, num_epochs, train_loader, device)

        # Test the model
        test_model(model, test_loader, device, threshold=0.5)


def bert_k_fold_cross_validation(batch_size, vector_size, learning_rate, num_epochs, chosen_label, device, embedding_model, k=5):
    # Load the annotated dialogues and labels
    annotated_dialogues, annotated_labels = data_prep()

    dialogue_max_length = max(len(dialogue) for dialogue in annotated_dialogues)
    sentence_max_length = 128

    # Split the data into k folds
    fold_size = int(len(annotated_dialogues) / k)
    for i in range(k):
        print("Fold: ", i+1)
        # Split the data into training and testing sets
        train_dialogues = annotated_dialogues[:i * fold_size] + annotated_dialogues[(i + 1) * fold_size:]
        train_labels = annotated_labels[:i * fold_size] + annotated_labels[(i + 1) * fold_size:]
        test_dialogues = annotated_dialogues[i * fold_size:(i + 1) * fold_size]
        test_labels = annotated_labels[i * fold_size:(i + 1) * fold_size]

        # Create the dialogue images
        train_dialogue_images_tensor = bert_create_dialogue_image(train_dialogues, dialogue_max_length, sentence_max_length)
        test_dialogue_images_tensor = bert_create_dialogue_image(test_dialogues, dialogue_max_length, sentence_max_length)

        # Create the label tensor
        train_label_tensor = create_label_tensor(train_labels, chosen_label, dialogue_max_length)
        test_label_tensor = create_label_tensor(test_labels, chosen_label, dialogue_max_length)

        # Create the datasets
        train_dataset = CasinoDataset(train_dialogue_images_tensor, train_label_tensor)
        test_dataset = CasinoDataset(test_dialogue_images_tensor, test_label_tensor)

        # Create the data loaders
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

        # Initialize the model
        model = ConvNet().to(device)

        # Train the model
        model = forward_pass(model, learning_rate, num_epochs, train_loader, device)

        # Test the model
        test_model(model, test_loader, device, threshold=0.5)





def main():
    # labels = ["small-talk", "showing-empathy", "promote-coordination", "no-need", "elicit-pref", "uv-part", "vouch-fair", "self-need", "other-need", "non-strategic"]

    batch_size = 16
    vector_size = 64
    learning_rate = 0.001
    num_epochs = 500
    chosen_label = "uv-part"
    embedding_model = "word2vec" # bert_base, word2vec, roberta

    print("Batch Size:", batch_size)
    print("Learning Rate:", learning_rate)
    print("Number of Epochs:", num_epochs)
    print("Chosen Label:", chosen_label)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    k_fold_cross_validation(batch_size, vector_size, learning_rate, num_epochs, chosen_label, device, embedding_model, k=5)




main()