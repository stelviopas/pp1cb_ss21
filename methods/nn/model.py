import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class FeedForwardNet(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()

        # set hyperparams
        self.hparams = hparams

        self.model = None

        ########################################################################
        # TODO: Initialize your model!                                         #
        ########################################################################

        embedding_dim = 1024
        longest_seq = 23
        input_size = longest_seq * embedding_dim
        output_size = longest_seq

        self.model = nn.Sequential(
            nn.Linear(input_size, self.hparams['hidden_size']),
            nn.ReLU(),
            nn.Linear(self.hparams['hidden_size'], output_size)
        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        # x.shape =  [batch_size, seq_length, 1024] -> flatten first

        x = x.view(x.shape[0], -1)

        # feed x into model
        x = self.model(x)

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


'''    def getTestAcc(self, loader = None):
        self.model.eval()
        self.model = self.model.to(self.device)

        if not loader: loader = self.test_dataloader()

        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X = X.to(self.device)
            score = self.forward(X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc'''


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
            #nn.Conv2d(1024, 60, (3, 3), stride=1, padding=2),
            nn.Conv2d(1024, 60, (3, 3)),
            nn.PReLU(),
            # nn.MaxPool2d(3),
            # nn.Conv2d(32, 64, (3, 3), stride=1, padding=2),
            # nn.PReLU(),
            # nn.MaxPool2d(2, stride=2),
            # nn.Conv2d(64, 128, (2, 2), stride=1, padding=1),
            # nn.PReLU(),
            # #Flatten(),
            # # issue here: linear layers require the most parameters
            # nn.Linear(10368, 256),
            # nn.Dropout(0.1),
            # nn.PReLU(),
            nn.Linear(256, 30)
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

        print("FORWARD PASS")
        print(x.size())
        x = self.model(x)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x

    def general_step(self, batch, batch_idx, mode):
        # forward pass
        out = self.forward(batch[0])

        # loss
        targets = batch[1]

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
