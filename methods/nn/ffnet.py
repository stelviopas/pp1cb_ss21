import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from methods.nn.disorder_dataset import load_dataset



class FFNet(pl.LightningModule):

    def __init__(self, hparams, *args, **kwargs):
        """
        Initialize your model from a given dict containing all your hparams
        """
        super().__init__()
        self.hparams = hparams
        self.hidden_size = hparams["hidden_size"]
        self.learning_rate = hparams["learning_rate"]
        self.window_size = hparams["window_size"]
        self.batch_size = hparams["batch_size"]
        self.test_results = 0.0
        self.truth_prediction_test = []

        ########################################################################
        # TODO: Define all the layers of your Feed Forward Network
        ########################################################################

        self.model = nn.Sequential(
            # input: batch_size * 1024 * window_size
            nn.Flatten(),
            # here, we have 1024 * window_size
            nn.Linear(1024 * self.window_size, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, 32),
            nn.Sigmoid(),
            nn.Linear(32, 1),
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

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "val")
        tensorboard_logs = {'val_loss': loss}
        return {'val_loss': loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "test")
        tensorboard_logs = {'test_loss': loss}

        # Remember raw ground truth and raw predictions 
        
        out = self.forward(batch['embeddings'])
        out = out.flatten()

        targets = batch['z_scores']
        self.truth_prediction_test.append([out, targets])
        
        return {'test_loss': loss, 'log': tensorboard_logs}


    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        return avg_loss
    
    def test_end(self, outputs):
        avg_loss = self.general_end(outputs, "test")
        tensorboard_logs = {'avg_test_loss': avg_loss}
        self.test_results = avg_loss
        return {'test_loss': avg_loss, 
        'log': tensorboard_logs}
    
    def validation_end(self, outputs):
        avg_loss = self.general_end(outputs, "val")
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), self.hparams["learning_rate"])
        return optim
