import torch
from torch import nn
import pytorch_lightning as pl
from disorder_dataset import load_dataset
from nested_cross_validation import nested_cross_validation



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
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=7),
            nn.PReLU(),
            # at this point, the size of the sequence is 1024 * (window_size - kernelsize + 1)
            # 15 - 6 = 9
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3),
            nn.PReLU(),
            # here, it is 9 - 2 = 7
            # maybe add this in later
            # nn.MaxPool1d(3),
            nn.Flatten(),
            nn.Linear(1024 * 7, 1)
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


if __name__ == "__main__":
    hparams = {'hidden_size': 112,
               "learning_rate": 1e-4,
               'window_size': 15
               }
    dataset = load_dataset(path="../../../pp1cb_ss21/data", window_size=hparams['window_size'])
    nested_cross_validation(dataset, mode='evaluate', model=ConvNet(hparams=hparams))
