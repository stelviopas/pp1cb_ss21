import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from methods.nn.disorder_dataset import load_dataset
from methods.nn.nested_cross_validation import nested_cross_validation


class FFNet(pl.LightningModule):

    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        """
        super(FFNet, self).__init__()
        self.hparams = hparams
        ########################################################################
        # TODO: Define all the layers of your Feed Forward Network
        ########################################################################

        self.model = nn.Sequential(
            # input: batch_size * 1024 * window_size
            nn.Flatten(),
            # here, we have 1024 * window_size
            nn.Linear(1024 * hparams['window_size'], 128),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.Sigmoid(),
            nn.Linear(64, 1),
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
        out = out.flatten()

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


if __name__ == "__main__":
    hparams = {'hidden_size': 112,
               "learning_rate": 1e-4,
               'window_size': 15
               }
    dataset = load_dataset(path="../../../pp1cb_ss21/data", window_size=hparams['window_size'])
    nested_cross_validation(dataset, mode='evaluate', model=FFNet(hparams=hparams))
