from transformers_modules import *
from helpers import *

deep_learning = True

filename = 'SRB_S1_data.pkl'

train_val_test_ratio = [0.7, 0.15, 0.15]

params = {
            'nhead': [2, 4],
            'output_dim': [128],
            'num_encoder_layers': [2],
            'dropout': [0.1],
        }

data = load_data(filename, train_val_test_ratio[0], train_val_test_ratio[2])
dates = data[0].columns.str[:4].unique()

if deep_learning:

    params['input_dim'] = [len(data[0].columns) // len(dates)]
    params['num_classes'] = [len(np.unique(data[1]))]

combination_params = [dict(zip(params.keys(), values)) for values in itertools.product(*params.values())]

df_res = pd.DataFrame()
model = TransformerModel

meta_params = {'filename': filename, 'deep_learning': deep_learning, 'dates': dates}

for p in combination_params:

    df_res = hyperparameter_opt(model, p, data, df_res, **meta_params)
    df_res.to_csv(f'results/hypp_opt_{filename[:filename.rfind("_")]}_{re.sub("[^A-Z]", "", model.__name__)}.csv')


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