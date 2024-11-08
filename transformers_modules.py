import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import torch.optim as optim
from networkx import directed_configuration_model
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class ObservationEmbedding(nn.Module):
    # Radim embedding i positional encoding (sin & cos)
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


class TransformerModel(nn.Module):
    # Pravim trasnformer block, koji je zapravo samo encoder block (iliti samo encoder layer); ne treba mi decoder
    def __init__(self, input_dim, num_classes, nhead=2, num_encoder_layers=2, dim_feedforward=128, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.observation_embedding = ObservationEmbedding(input_dim, dim_feedforward)

        encoder_layers = TransformerEncoderLayer(d_model=dim_feedforward, nhead=nhead, dim_feedforward=dim_feedforward,
                                                 dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        # self.decoder = nn.Linear(dim_feedforward, num_classes)

    def forward(self, src, dates):
        src = self.observation_embedding(src, dates)
        output = self.transformer_encoder(src)
        # output = output.mean(dim=1)  # Average pooling
        # output = self.decoder(output)
        # print(f"After TransformerModel, shape is: {output.shape}")
        return output

class Classifier(nn.Module):
    # Transformet je bio feature extractor, sad pravim slassifier
    # Ovde mi treba MaxPool ili AvgPool, potom FC_lin_layer, i na kraju SoftMax

    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)  # Global Average Pooling
        self.fc = nn.Linear(input_dim, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Apply pooling layer
        x = self.pool(x)  # Shape: (batch_size, 1, input_dim)
        # print(f"After AVGPOOL, shape is: {x.shape}")
        # Squeeze the dimension
        x = x.squeeze(2)  # Shape: (batch_size, input_dim)
        # print(f"After squeeze, shape is: {x.shape}")
        # Apply FC layer
        x = self.fc(x)
        # print(f"After FC, shape is: {x.shape}")
        # Apply softmax for probability distribution
        output = x
        # output = self.softmax(x)
        return output

# Napravio sam CombinedModel da bih mogao kasnije da freezujem feature extractor (tj. TransformerModel) i treniram samo Classifier head.
# Ali nesto ne valja sa dimenzijama, tj. shape-ovima kad radim matmul u FC-layeru. To moras nekako da isproveravas.
# Najverovatnije kombinovanje FeatureExtractor dela sa Classifier head nesto zeza!!!!!
# Kako proveriti dimenzije nakon prolaska kroz nekoliko hidden lejera? Nesto interaktivno, nesto jednostavno?

class CombinedModel(nn.Module):
    def __init__(self, feature_extractor, classifier):
        super(CombinedModel, self).__init__()
        # Use externally provided feature extractor and classifier
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, src, dates):
        # Pass through the feature extractor
        features = self.feature_extractor(src, dates)
        # print(f"In CombinedModel, after feature extractor, shape is: {features.shape}")
        # Classifier head for final prediction
        output = self.classifier(features)
        # print(f"In CombinedModel, after classifier, shape is: {output.shape}")
        return output

    def freeze_feature_extractor(self):
        # Freeze all parameters in the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def unfreeze_feature_extractor(self):
        # Unfreeze all parameters in the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = True

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(13*3, 16)  # First fully connected layer
        self.fc2 = nn.Linear(16, 8)   # Second fully connected layer
        self.fc3 = nn.Linear(8, 1)    # Output layer for 2 classes

    def forward(self, x):
        # print(f"Before 1st FC layer x.shape is: {x.shape}")
        x = torch.relu(self.fc1(x))
        # print(f"After 1st FC layer x.shape is: {x.shape}")
        x = torch.relu(self.fc2(x))
        # print(f"After 2nd FC layer x.shape is: {x.shape}")
        x = self.fc3(x)
        # print(f"After 3rd FC layer x.shape is: {x.shape}")
        return x