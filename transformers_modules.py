import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class ObservationEmbedding(nn.Module):
    def __init__(self, feature_dim, embedding_dim):
        super(ObservationEmbedding, self).__init__()
        self.feature_embed = nn.Linear(feature_dim, embedding_dim // 2)
        self.date_embed_dim = embedding_dim // 2
        self.embedding_dim = embedding_dim

    def positional_encoding(self, dates, batch_size):
        start_date = datetime.strptime(min(dates), "%m%d")
        positions = [(datetime.strptime(date, "%m%d") - start_date).days for date in dates]

        pos_encoding = np.zeros((len(positions), self.date_embed_dim))
        for pos, day in enumerate(positions):
            for i in range(0, self.date_embed_dim, 2):
                pos_encoding[pos, i] = np.sin(day / (10000 ** (2 * i / self.date_embed_dim)))
                pos_encoding[pos, i + 1] = np.cos(day / (10000 ** (2 * i / self.date_embed_dim)))

        pos_encoding = torch.tensor(pos_encoding, dtype=torch.float32)
        pos_encoding = pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: (batch_size, time_steps, date_embed_dim)
        return pos_encoding

    def forward(self, features, dates):
        batch_size, time_steps, _ = features.shape

        feature_emb = self.feature_embed(features)  # Shape: (batch_size, time_steps, feature_embed_dim)

        date_positional_emb = self.positional_encoding(dates, batch_size).to(features.device)  # Shape: (batch_size, time_steps, date_embed_dim)

        observation_emb = torch.cat((feature_emb, date_positional_emb),dim=-1)  # Shape: (batch_size, time_steps, embedding_dim)
        return observation_emb


class EncoderBlock(nn.Module):
    def __init__(self, nhead=2, output_dim=128, num_encoder_layers=2, dropout=0.2, input_dim=3):
        super(EncoderBlock, self).__init__()
        self.model_type = 'Transformer'
        self.observation_embedding = ObservationEmbedding(input_dim, output_dim)

        encoder_layers = TransformerEncoderLayer(d_model=output_dim, nhead=nhead, dim_feedforward=output_dim, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)

    def forward(self, src, dates):
        src = self.observation_embedding(src, dates)
        output = self.transformer_encoder(src)
        return output

class Classifier(nn.Module):
    def __init__(self, input_dim=128, num_classes=9):
        super(Classifier, self).__init__()

        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = x.mean(dim=1)  # Average pooling
        output = self.fc(x)

        return output

class TransformerModel(nn.Module):
    def __init__(self, nhead, output_dim, num_encoder_layers, dropout, input_dim, num_classes):

        super(TransformerModel, self).__init__()
        self.feature_extractor = EncoderBlock(nhead, output_dim, num_encoder_layers, dropout, input_dim)
        self.classifier = Classifier(output_dim, num_classes)

    def fit(self, model, train_loader, val_loader, dates, num_epochs=25):
        best_model_wts = model.state_dict()
        best_acc = 0.0

        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            running_corrects = 0
            print(epoch)
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs, dates)
                loss = criterion(torch.squeeze(outputs), labels)
                loss.backward()
                optimizer.step()
                _, preds = torch.max(outputs, 1)
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
                    outputs = model(inputs, dates)
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

    def predict(self, model, dataloader, dates):
        model.eval()

        predictions = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs, dates)
                _, preds = torch.max(outputs, 1)

                predictions.extend(preds)

        return predictions

    def forward(self, src, dates):
        features = self.feature_extractor(src, dates)
        output = self.classifier(features)

        return output

    def freeze_feature_extractor(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def unfreeze_feature_extractor(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = True

class TransformerClassifier(TransformerModel):

    def __init__(self,
        nhead = 2,
        output_dim = 128,
        num_encoder_layers=2,
        dropout = 0.1,
        input_dim = None,
        num_classes = 9,
                 ):
        super().__init__(
            nhead,
            output_dim,
            num_encoder_layers,
            dropout,
            input_dim,
            num_classes,
            ),

        self.nhead = nhead
        self.output_dim = output_dim
        self.num_encoder_layers = num_encoder_layers
        self.dropout = dropout
        self.input_dim = input_dim
        self.num_classes = num_classes



