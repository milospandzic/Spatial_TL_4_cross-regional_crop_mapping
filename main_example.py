from transformers_modules import *
from helpers import *

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Load the data
filename = 'SRB_S1_data.pkl'

# Preprocess data to create sequences

train_val_test_ratio = [0.7, 0.15, 0.15]

data = load_data(filename, train_val_test_ratio[0], train_val_test_ratio[2])

X_train, Y_train, X_val, Y_val, _, Y_test = data

dates = X_train.columns.str[:4].unique()

X_train = reshape_input(X_train)
X_val = reshape_input(X_val)

train_dataset = TimeSeriesDataset(X_train, Y_train.values)
val_dataset = TimeSeriesDataset(X_val, Y_val.values)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

params = {
        'feature_extractor_params': {
            'nhead': [2, 4],
            'dim_feedforward': [128],
            'num_encoder_layers': [2],
            'dropout': [0.1],
            'input_dim': [X_train.shape[2]],
            'num_classes': [len(np.unique(Y_train))]}
    ,
        'classifier_params': {
            'input_dim': 128,
             'num_classes': len(np.unique(Y_train))
        },

}

combination_params = [dict(zip(params['feature_extractor_params'].keys(), values)) for values in itertools.product(*params['feature_extractor_params'].values())]

model_def = TransformerModel

df_res = pd.DataFrame()

for p in combination_params:

    params['feature_extractor_params'] = p

    # df_res = hyperparameter_opt(model, params, data, df_res, filename)

    # feature_extractor = EncoderBlock(**p)
    # classifier = Classifier(input_dim=128, num_classes=len(np.unique(Y_train)))
    # model = TransformerModel(feature_extractor=feature_extractor, classifier=classifier)

    model = model_def(**params)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    model = model.fit(model, train_loader, val_loader, criterion, optimizer, dates, num_epochs=3)

# # Save the model weights and optimizer state
# def save_model(model, optimizer, epoch, path="transformer_model.pth"):
#     torch.save({
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#     }, path)
#     print(f"Model saved to {path}")
#
#
# # Example usage during/after training
# save_model(model, optimizer, epoch=3, path="trained_transformer_model.pth")
#
#
# def evaluate_model(model, dataloader):
#     model.eval()
#     corrects = 0
#     total = 0
#
#     with torch.no_grad():
#         for inputs, labels in dataloader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs, dates)
#             _, preds = torch.max(outputs, 1)
#             corrects += torch.sum(preds == labels.data)
#             total += labels.size(0)
#
#     accuracy = corrects.double() / total
#     return accuracy.item()
#
#
# test_accuracy = evaluate_model(model, test_loader)
# print("SLO Test Accuracy:", test_accuracy)
#
# srb_accuracy = evaluate_model(model, test_loader_srb)
# print("SRB Test Accuracy:", srb_accuracy)
#
# model.freeze_feature_extractor()
# modelSRB = train_model(model, train_loader_srb, val_loader_srb, criterion, optimizer, num_epochs=3)
# TLsrb_accuracy = evaluate_model(modelSRB, test_loader_srb)
# print("Transfer Learning - SRB Test Accuracy:", TLsrb_accuracy)