import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from ptdec.dec import DEC
from ptdec.model import predict
from ptsdae.sdae import StackedDenoisingAutoEncoder
from ptdec.utils import cluster_accuracy
import uuid
import click

class ReutersDataset(Dataset):
    def __init__(self, features, labels, cuda):
        self.features = features
        self.labels = labels
        self.cuda = cuda

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        feature = torch.tensor(feature, dtype=torch.float)
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
    "--target-cluster",
    help="the target cluster to get top scoring elements from (default 0).",
    type=int,
    default=0
)
def main(cuda, target_cluster):
    # Load the Reuters dataset
    dataset = load_dataset('reuters21578', 'ModHayes')
    texts = [item['text'] for item in dataset['train']] + [item['text'] for item in dataset['test']]
    labels = [item['label'] for item in dataset['train']] + [item['label'] for item in dataset['test']]
    
    # Convert texts to numerical features
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=2000)
    features = vectorizer.fit_transform(texts).toarray()
    
    # Split the features back into train and test sets
    train_size = len(dataset['train'])
    features_train = features[:train_size]
    labels_train = labels[:train_size]
    features_test = features[train_size:]
    labels_test = labels[train_size:]
    
    ds_train = ReutersDataset(features=features_train, labels=labels_train, cuda=cuda)
    
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

    # Generate the confusion matrix plot
    predicted_reassigned = [
        reassignment[item] for item in predicted
    ]
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
