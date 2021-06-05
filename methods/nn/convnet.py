import os
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data.dataset import Dataset
from disorder_dataset import load_dataset
import numpy as np


class ConvNet(pl.LightningModule):
    """Facial keypoint detection model"""

    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the  method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
        """
        super(ConvNet, self).__init__()
        self.hparams = hparams
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        ########################################################################

        self.model = nn.Sequential(
            # input: batch_size * 1024 * window_size
            nn.Conv2d(in_channels=1024, out_channels=64, kernel_size=5),
            nn.PReLU(),
            #nn.Conv1d(in_channels=64, out_channels=16, kernel_size=(3, 1024)),
            #nn.PReLU(),
            # nn.MaxPool1d(3),
            nn.Flatten(),
            # formula:
            # shape here: 100 (batch_size) * 144 (16*3*3)
            nn.Linear(96, 1)
        )

        print(self.model)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints                                    #
        ########################################################################
        print(x.shape)
        x = self.model(x)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x

    def general_step(self, batch, batch_idx, mode):
        # forward pass
        out = self.forward(batch['embeddings'])

        # loss
        targets = batch['z_scores']

        # print(out.shape)
        # print(targets.shape)

        loss = nn.MSELoss()
        loss = loss(out, targets)

        return loss

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        return avg_loss

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {'loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "val")
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss}

    def validation_end(self, outputs):
        avg_loss = self.general_end(outputs, "val")
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), self.hparams["learning_rate"])
        return optim


def nested_cross_validation(dataset,
                            model=None,
                            mode='print_fold_info',
                            k=10, *kwargs):
    # Set fixed random number seed
    SEED = 1
    torch.manual_seed(SEED)

    # Define the K-fold Cross Validator
    skf = StratifiedKFold(n_splits=k, random_state=SEED, shuffle=True)

    # For folds results
    results = {}
    # For debugging loaders
    loaders = {}

    # Nested K-Fold Cross Validation model evaluation
    # We split the data stratified on artificially constructed bins in df['bins'
    # and extract indices.

    # By splitting we only extract indices of samples for each test/train/val sets,
    # thus we only need either X or y (equal length).
    # For stratification, however, we require the artificially assigned bins whic are also defined in the
    # dataset class

    data = dataset.y
    stratify_on = dataset.bins

    for fold, (train_val_ids, test_ids) in enumerate(skf.split(data, stratify_on)):

        print(f"Fold {fold}")

        train_ids, val_ids = train_test_split(train_val_ids,
                                              test_size=0.20,  # 0.25 x 0.8 = 0.2
                                              stratify=stratify_on[train_val_ids],
                                              random_state=SEED)

        # Define data loaders for training and testing data in this fold
        valloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=100,
            #collate_fn=collate,
            sampler=val_ids)

        trainloader = torch.utils.data.DataLoader(
            dataset,
            #collate_fn=collate,
            batch_size=100, sampler=train_ids)

        testloader = torch.utils.data.DataLoader(
            dataset,
            #collate_fn=collate,
            batch_size=100, sampler=test_ids)

        if mode == 'print_fold_info':

            loaders[fold] = [trainloader, valloader, testloader]

            print('train -  {}, avg_len - {:.2f}  |  val -  {}, avg_len - {:.2f}'.format(
                np.sum(np.bincount(stratify_on[train_ids])),
                np.sum(dataset.aa_len[train_ids]) / len(train_ids),
                np.sum(np.bincount(stratify_on[val_ids])),
                np.sum(dataset.aa_len[val_ids]) / len(val_ids)))

            print('test -  {}, avg_len - {:.2f}'.format(
                np.sum(np.bincount(stratify_on[test_ids])),
                np.sum(dataset.aa_len[test_ids]) / len(test_ids)))

            show_batches = 5
            for i, batch in enumerate(trainloader):
                print(f"Batch {i}:\n{batch}\n")
                if i == show_batches:
                    break

            print()

        elif mode == 'evaluate':
            trainer = pl.Trainer(weights_summary=None, max_epochs=100, deterministic=True)
            trainer.fit(model, train_dataloader=trainloader, val_dataloaders=valloader)
        else:
            print("Mode is not specified!")
            break

    if mode == 'evaluate':
        return results
    else:
        return loaders


if __name__ == "__main__":
    hparams = {'hidden_size': 112,
               "learning_rate": 1e-4,
               'window_size': 15
               }
    dataset = load_dataset(path="/home/matthias/Code/Python/pp1cb_ss21/data", window_size=hparams['window_size'])

    nested_cross_validation(dataset, mode='evaluate', model=ConvNet(hparams=hparams))
