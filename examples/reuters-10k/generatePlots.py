import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import Dataset
import scipy.io as sio
from ptdec.dec import DEC
from ptdec.model import predict
from ptsdae.sdae import StackedDenoisingAutoEncoder
from ptdec.utils import cluster_accuracy
import uuid
import click

class ReutersDataset(Dataset):
    def __init__(self, features, labels, cuda):
        self.features = features
        self.labels = labels.squeeze()
        self.cuda = cuda

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        feature = torch.tensor(feature, dtype=torch.float)  # Convert to PyTorch tensor
        label = torch.tensor(label, dtype=torch.long)
        if self.cuda:
            feature = feature.cuda()
            label = label.cuda()
        return feature, label

def load_models(cuda):
    autoencoder = StackedDenoisingAutoEncoder(
        [2000, 500, 500, 2000, 10], final_activation=None
    )
    autoencoder.load_state_dict(torch.load("finetuned_model.pth"))
    if cuda:
        autoencoder.cuda()

    model = DEC(cluster_number=10, hidden_dimension=10, encoder=autoencoder.encoder)
    model.load_state_dict(torch.load("dec_model.pth"))
    if cuda:
        model.cuda()
    
    return autoencoder, model

def generate_confusion_matrix(actual, predicted_reassigned, class_range):
    confusion = confusion_matrix(actual, predicted_reassigned, labels=class_range)
    
    # Normalize confusion matrix
    confusion_normalized = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
    confusion_normalized[np.isnan(confusion_normalized)] = 0  # replace NaN with 0
    
    # Plotting
    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_normalized, annot=True, cmap='magma', fmt='.2f', cbar=True)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(np.arange(len(class_range)) + 0.5, [f'Class {i}' for i in class_range], rotation=45)
    plt.yticks(np.arange(len(class_range)) + 0.5, [f'Class {i}' for i in class_range], rotation=45)
    
    # Save the figure
    confusion_id = uuid.uuid4().hex
    plt.savefig("confusion_%s.png" % confusion_id)
    plt.show()

    print("Writing out confusion diagram with UUID: %s" % confusion_id)

@click.command()
@click.option(
    "--cuda", 
    help="Whether to use CUDA for GPU acceleration.",
    type=bool, 
    default=False
)
@click.option(
    "--mat-file",
    help="Path to the reuters10k.mat file.",
    type=str,
    default="examples/reuters_10k/reuters10k.mat"
)
@click.option(
    "--class-range",
    help="Range of classes to include in the confusion matrix.",
    type=str,
    default="0,1,2,3"
)
def main(cuda, mat_file, class_range):
    class_range = list(map(int, class_range.split(',')))
    
    mat_contents = sio.loadmat(mat_file)
    features = mat_contents['X']
    labels = mat_contents['Y']

    ds_train = ReutersDataset(features=features, labels=labels, cuda=cuda)  # training dataset

    autoencoder, model = load_models(cuda)

    # Generate predictions
    predicted, actual = predict(
        ds_train, model, 1024, silent=True, return_actual=True, cuda=cuda
    )
    actual = actual.cpu().numpy()
    predicted = predicted.cpu().numpy()
    reassignment, accuracy = cluster_accuracy(actual, predicted)
    print("Final DEC accuracy: %s" % accuracy)

    # Generate the confusion matrix plot
    predicted_reassigned = [
        reassignment[item] for item in predicted
    ]
    generate_confusion_matrix(actual, predicted_reassigned, class_range)

if __name__ == "__main__":
    main()
