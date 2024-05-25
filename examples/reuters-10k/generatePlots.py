import numpy as np
import seaborn as sns
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


@click.command()
@click.option(
    "--cuda", help="whether to use CUDA (default False).", type=bool, default=False
)
@click.option(
    "--mat-file",
    help="path to the reuters10k.mat file.",
    type=str,
    default="examples/reuters_10k/reuters10k.mat"
)
@click.option(
    "--target-cluster",
    help="the target cluster to get top scoring elements from (default 0).",
    type=int,
    default=0
)
def main(cuda, mat_file, target_cluster):
    mat_contents = sio.loadmat(mat_file)
    features = mat_contents['X']
    labels = mat_contents['Y']

    ds_train = ReutersDataset(features=features, labels=labels, cuda=cuda)  # training dataset

    # Load the pretrained autoencoder
    autoencoder = StackedDenoisingAutoEncoder(
        [2000, 500, 500, 2000, 10], final_activation=None
    )
    autoencoder.load_state_dict(torch.load("finetuned_model.pth"))
    if cuda:
        autoencoder.cuda()

    # Load the trained DEC model
    model = DEC(cluster_number=10, hidden_dimension=10, encoder=autoencoder.encoder)
    model.load_state_dict(torch.load("dec_model.pth"))
    if cuda:
        model.cuda()

    # Generate predictions
    predicted, actual = predict(
        ds_train, model, 1024, silent=True, return_actual=True, cuda=cuda
    )
    actual = actual.cpu().numpy()
    predicted = predicted.cpu().numpy()
    reassignment, accuracy = cluster_accuracy(actual, predicted)
    print("Final DEC accuracy: %s" % accuracy)

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

    # Print the matrix
    print("Top 10 scoring elements in each cluster:")
    print(top_elements_matrix)
    # for cluster in range(10):
    #     # Get top 10 scoring elements from the current cluster
    #     top_indices = np.argsort(q[:, cluster])[-10:]
    #     top_scores = q[top_indices, cluster]

    #     print(f"Top 10 scoring elements in cluster {cluster}:")
    #     for idx, score in zip(top_indices, top_scores):
    #         # Here you can print something else instead of just the indices
    #         print(f"Element: {idx}, Score: {score}")


    # Generate the confusion matrix plot
    predicted_reassigned = [
        reassignment[item] for item in predicted
    ]  # TODO numpify
    confusion = confusion_matrix(actual, predicted_reassigned)
    normalised_confusion = (
        confusion.astype("float") / confusion.sum(axis=1)[:, np.newaxis]
    )
    confusion_id = uuid.uuid4().hex
    sns.heatmap(normalised_confusion).get_figure().savefig(
        "confusion_%s.png" % confusion_id
    )
    print("Writing out confusion diagram with UUID: %s" % confusion_id)


if __name__ == "__main__":
    main()
