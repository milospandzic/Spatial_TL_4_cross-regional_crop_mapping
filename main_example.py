import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from transformers_modules import ObservationEmbedding,TransformerModel,Classifier,CombinedModel, SimpleNN
import torch.optim as optim

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import LabelEncoder

# Load the data
slo_data = pd.read_pickle('data/SLO_S1_data.pkl')
srb_data = pd.read_pickle('data/SRB_S1_data.pkl')
dates = slo_data.drop(columns=['ID', 'class']).columns.str[:4].unique()


# Preprocess data to create sequences
def preprocess_data(df):
    df = df.sort_index(axis=1)  # Ensure columns are sorted by time
    X = df.drop(columns=['ID', 'class']).values.reshape(-1, len(df.columns) // 3,
                                                        3)  # Reshape to (samples, time_steps, features)
    y = LabelEncoder().fit_transform(df['class'].values)  # Encode labels to integers
    return X, y


X_slo, y_slo = preprocess_data(slo_data)
X_srb, y_srb = preprocess_data(srb_data)

# Split the SLO data into training (70%), validation (15%), and testing (15%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X_slo, y_slo, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Split the SRB data into training (15%), validation (15%), and testing (70%) sets
X_temp_srb, X_test_srb, y_temp_srb, y_test_srb = train_test_split(X_srb, y_srb, test_size=0.7, random_state=42)
X_train_srb, X_val_srb, y_train_srb, y_val_srb = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = TimeSeriesDataset(X_train, y_train)
val_dataset = TimeSeriesDataset(X_val, y_val)
test_dataset = TimeSeriesDataset(X_test, y_test)

train_dataset_srb = TimeSeriesDataset(X_train_srb, y_train_srb)
val_dataset_srb = TimeSeriesDataset(X_val_srb, y_val_srb)
test_dataset_srb = TimeSeriesDataset(X_test_srb, y_test_srb)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

train_loader_srb = DataLoader(train_dataset_srb, batch_size=32, shuffle=False)
val_loader_srb = DataLoader(val_dataset_srb, batch_size=32, shuffle=False)
test_loader_srb = DataLoader(test_dataset_srb, batch_size=32, shuffle=False)

input_dim = X_train.shape[2]
num_classes = len(np.unique(y_slo))
dim_feedforward = 13

# Initialize the feature extractor and classifier separately
feature_extractor = TransformerModel(input_dim=input_dim, num_classes=num_classes)
classifier = Classifier(input_dim=dim_feedforward, num_classes=num_classes)
# Initialize the combined model
model = CombinedModel(feature_extractor=feature_extractor, classifier=classifier)
# model = SimpleNN()


# model = TransformerModel(input_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
model.to(device)


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        print(epoch)
        for inputs, labels in train_loader:
            # print(inputs.shape, inputs)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # outputs = model(torch.reshape(inputs, (-1, 13 * 3)))
            outputs = model(inputs, dates)
            # print(np.shape(outputs), np.shape(labels))
            # outputs = model(inputs, dates)
            # loss = criterion(outputs, torch.unsqueeze(labels, 1))
            loss = criterion(torch.squeeze(outputs), labels)
            # print(loss)
            loss.backward()
            optimizer.step()
            # print(outputs.shape, outputs)
            _, preds = torch.max(outputs, 1)
            # print(preds)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f'Epoch {epoch}/{num_epochs - 1} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                # outputs = model(torch.reshape(inputs, (-1, 13 * 3)))
                outputs = model(inputs, dates)
                # loss = criterion(outputs, torch.unsqueeze(labels, 1))
                loss = criterion(torch.squeeze(outputs), labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)

        print(f'Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()

    model.load_state_dict(best_model_wts)
    return model


model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25)


# Save the model weights and optimizer state
def save_model(model, optimizer, epoch, path="transformer_model.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Model saved to {path}")


# Example usage during/after training
save_model(model, optimizer, epoch=25, path="trained_transformer_model.pth")


def evaluate_model(model, dataloader):
    model.eval()
    corrects = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs, dates)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

    accuracy = corrects.double() / total
    return accuracy.item()


test_accuracy = evaluate_model(model, test_loader)
print("SLO Test Accuracy:", test_accuracy)

srb_accuracy = evaluate_model(model, test_loader_srb)
print("SRB Test Accuracy:", srb_accuracy)
