import click
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import torch
from torch.utils.data import Dataset
from tensorboardX import SummaryWriter
import uuid
import os
import scipy.io as sio

from ptdec.dec import DEC
from ptdec.model import train, predict, target_distribution
from ptsdae.sdae import StackedDenoisingAutoEncoder
import ptsdae.model as ae
from ptdec.utils import cluster_accuracy
from sklearn.model_selection import train_test_split


class ReutersDataset(Dataset):
    def __init__(self, features, labels, cuda):
        self.features = torch.tensor(features, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.long)
        if cuda:
            self.features = self.features.cuda()
            self.labels = self.labels.cuda()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label


@click.command()
@click.option(
    "--cuda", help="whether to use CUDA (default False).", type=bool, default=False
)
@click.option(
    "--batch-size", help="training batch size (default 256).", type=int, default=256
)
@click.option(
    "--pretrain-epochs",
    help="number of pretraining epochs (default 300).",
    type=int,
    default=300,
)
@click.option(
    "--finetune-epochs",
    help="number of finetune epochs (default 500).",
    type=int,
    default=500,
)
@click.option(
    "--testing-mode",
    help="whether to run in testing mode (default False).",
    type=bool,
    default=False,
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
def main(cuda, batch_size, pretrain_epochs, finetune_epochs, testing_mode, mat_file, target_cluster):
    writer = SummaryWriter()  # create the TensorBoard object

    def training_callback(epoch, lr, loss, validation_loss):
        writer.add_scalars(
            "data/autoencoder",
            {"lr": lr, "loss": loss, "validation_loss": validation_loss,},
            epoch,
        )

    mat_contents = sio.loadmat(mat_file)
    features = mat_contents['X']
    labels = mat_contents['Y'].squeeze()  # Ensure labels are 1D

    # Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

    ds_train = ReutersDataset(features=X_train, labels=y_train, cuda=cuda)  # training dataset
    ds_val = ReutersDataset(features=X_val, labels=y_val, cuda=cuda)  # validation dataset

    autoencoder = StackedDenoisingAutoEncoder(
        [2000, 500, 500, 2000, 10], final_activation=None
    )
    if cuda:
        autoencoder.cuda()
    print("Pretraining stage.")
    ae.pretrain(
        ds_train,
        autoencoder,
        cuda=cuda,
        validation=ds_val,
        epochs=pretrain_epochs,
        batch_size=batch_size,
        optimizer=lambda model: SGD(model.parameters(), lr=0.1, momentum=0.9),
        scheduler=lambda x: StepLR(x, 100, gamma=0.1),
        corruption=0.2,
    )
    # Save the pretrained model
    torch.save(autoencoder.state_dict(), "pretrained_model.pth")

    print("Training stage.")
    ae_optimizer = SGD(params=autoencoder.parameters(), lr=0.1, momentum=0.9)
    ae.train(
        ds_train,
        autoencoder,
        cuda=cuda,
        validation=ds_val,
        epochs=finetune_epochs,
        batch_size=batch_size,
        optimizer=ae_optimizer,
        scheduler=StepLR(ae_optimizer, 100, gamma=0.1),
        corruption=0.2,
        update_callback=training_callback,
    )
    # Save the fine-tuned model
    torch.save(autoencoder.state_dict(), "finetuned_model.pth")

    print("DEC stage.")
    model = DEC(cluster_number=10, hidden_dimension=10, encoder=autoencoder.encoder)
    if cuda:
        model.cuda()
    dec_optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    train(
        dataset=ds_train,
        model=model,
        epochs=100,
        batch_size=256,
        optimizer=dec_optimizer,
        stopping_delta=0.000001,
        cuda=cuda,
    )
    # Save the DEC model
    torch.save(model.state_dict(), "dec_model.pth")

    predicted, actual = predict(
        ds_val, model, 1024, silent=True, return_actual=True, cuda=cuda  # Dùng tập validation để kiểm tra
    )
    actual = actual.cpu().numpy()
    predicted = predicted.cpu().numpy()
    reassignment, accuracy = cluster_accuracy(actual, predicted)
    print("Final DEC accuracy: %s" % accuracy)

    # Get the soft assignments
    model.eval()
    with torch.no_grad():
        q = model(ds_val.features)
        if cuda:
            q = q.cpu()
        q = q.numpy()

    # Get top 10 scoring elements from each cluster
    top_elements_matrix = np.zeros((10, 10), dtype=int)

    for cluster in range(10):
        top_indices = np.argsort(q[:, cluster])[-10:]
        top_elements_matrix[cluster] = top_indices

    print("Top 10 scoring elements in each cluster:")
    for cluster in range(10):
        print(f"\nCluster {cluster}:")
        for idx in top_elements_matrix[cluster]:
            score = q[idx, cluster]
            print(f"Index {idx}: Score {score}")

    if not testing_mode:
        predicted_reassigned = [
            reassignment[item] for item in predicted
        ]
        confusion = confusion_matrix(actual, predicted_reassigned)
        
        # Handling division by zero
        confusion_sum = confusion.sum(axis=1)[:, np.newaxis]
        confusion_sum[confusion_sum == 0] = 1
        normalised_confusion = confusion.astype("float") / confusion_sum
        
        confusion_id = uuid.uuid4().hex
        sns.heatmap(normalised_confusion).get_figure().savefig(
            "confusion_%s.png" % confusion_id
        )
        print("Writing out confusion diagram with UUID: %s" % confusion_id)
        writer.close()


if __name__ == "__main__":
    main()
