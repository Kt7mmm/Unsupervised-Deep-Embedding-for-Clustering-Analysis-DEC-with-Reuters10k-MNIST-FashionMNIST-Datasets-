import numpy as np
import torch
import scipy.io as sio
from torch.utils.data import Dataset
from ptdec.dec import DEC
from ptdec.model import predict
from ptsdae.sdae import StackedDenoisingAutoEncoder
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
def main(cuda, mat_file):
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

    # Get the soft assignments
    model.eval()
    with torch.no_grad():
        q = model(torch.tensor(ds_train.features, dtype=torch.float).cuda() if cuda else torch.tensor(ds_train.features, dtype=torch.float))
        if cuda:
            q = q.cpu()
        q = q.numpy()

    # Iterate over each cluster
    # Initialize an empty matrix to store the top elements
    top_elements_matrix = np.zeros((10, 10), dtype=int)

    # Iterate over each cluster
    for cluster in range(10):
        # Get top 10 scoring elements from the current cluster
        top_indices = np.argsort(q[:, cluster])[-10:]
        
        # Store the top indices in the matrix
        top_elements_matrix[cluster] = top_indices

    # Print the top scoring elements with their data
    print("Top 10 scoring elements in each cluster:")
    for cluster in range(10):
        print(f"\nCluster {cluster}:")
        for idx in top_elements_matrix[cluster]:
            document_content = features[idx]  # Assuming features are document vectors
            print(f"Index {idx}: {document_content[:100]}")  # Print the first 100 characters

if __name__ == "__main__":
    main()
