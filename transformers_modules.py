import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ObservationEmbedding(nn.Module):
    # In charge of embedding i positional encoding (sin & cos)
    def __init__(self, feature_dim, embedding_dim):
        super(ObservationEmbedding, self).__init__()
        self.feature_embed = nn.Linear(feature_dim, embedding_dim // 2)
        self.date_embed_dim = embedding_dim // 2
        self.embedding_dim = embedding_dim

    def positional_encoding(self, dates, batch_size):
        # Convert date strings to "days since start" for positional encoding
        start_date = datetime.strptime(min(dates), "%m%d")
        positions = [(datetime.strptime(date, "%m%d") - start_date).days for date in dates]

        # Generate the positional encoding using sin/cos functions
        pos_encoding = np.zeros((len(positions), self.date_embed_dim))
        for pos, day in enumerate(positions):
            for i in range(0, self.date_embed_dim, 2):
                pos_encoding[pos, i] = np.sin(day / (10000 ** (2 * i / self.date_embed_dim)))
                pos_encoding[pos, i + 1] = np.cos(day / (10000 ** (2 * i / self.date_embed_dim)))

        # Convert to tensor and add batch dimension
        pos_encoding = torch.tensor(pos_encoding, dtype=torch.float32)
        pos_encoding = pos_encoding.unsqueeze(0).expand(batch_size, -1,
                                                        -1)  # Shape: (batch_size, time_steps, date_embed_dim)
        return pos_encoding

    def forward(self, features, dates):
        batch_size, time_steps, _ = features.shape

        # Feature embedding
        feature_emb = self.feature_embed(features)  # Shape: (batch_size, time_steps, feature_embed_dim)

        # Date positional encoding
        date_positional_emb = self.positional_encoding(dates, batch_size).to(
            features.device)  # Shape: (batch_size, time_steps, date_embed_dim)

        # Concatenate feature and positional embeddings
        observation_emb = torch.cat((feature_emb, date_positional_emb),
                                    dim=-1)  # Shape: (batch_size, time_steps, embedding_dim)
        # print(f"After ObservationEmbedding, shape is: {observation_emb.shape}")
        return observation_emb


class EncoderBlock(nn.Module):
    # Pravim trasnformer block, koji je zapravo samo encoder block (iliti samo encoder layer); ne treba mi decoder
    def __init__(self, input_dim, num_classes, nhead=2, num_encoder_layers=2, dim_feedforward=128, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.model_type = 'Transformer'
        self.observation_embedding = ObservationEmbedding(input_dim, dim_feedforward)

        encoder_layers = TransformerEncoderLayer(d_model=dim_feedforward, nhead=nhead, dim_feedforward=dim_feedforward,
                                                 dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)

    def forward(self, src, dates):
        src = self.observation_embedding(src, dates)
        output = self.transformer_encoder(src)
        return output

class Classifier(nn.Module):
    # Transformer is Feature Extractor. Now defining Classifier.
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # Apply pooling layer
        x = x.mean(dim=1)  # Average pooling
        # Apply FC layer
        output = self.fc(x)

        return output

# TransformerModel generated in order to be able to freeze FeatureExtractor and unfreeze only Classifier when retraining.
class TransformerModel(nn.Module):
    def __init__(self, feature_extractor_params, classifier_params):
        super(TransformerModel, self).__init__()
        self.feature_extractor = EncoderBlock(**feature_extractor_params)
        self.classifier = Classifier(**classifier_params)


    def fit(self, model, train_loader, val_loader, criterion, optimizer, dates, num_epochs=25):
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

    def forward(self, src, dates):
        # Pass through the feature extractor
        features = self.feature_extractor(src, dates)
        # Pass through Classifier head for final prediction
        output = self.classifier(features)

        return output

    def freeze_feature_extractor(self):
        # Freeze all parameters in the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def unfreeze_feature_extractor(self):
        # Unfreeze all parameters in the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = True