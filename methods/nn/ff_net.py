import os
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl


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
            nn.Conv1d(in_channels=1024, out_channels=64, kernel_size=4),
            nn.PReLU(),
            nn.Conv1d(in_channels=64, out_channels=16, kernel_size=2),
            nn.PReLU(),
            # nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3),
            # nn.PReLU(),
            # nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3),
            # nn.PReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
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