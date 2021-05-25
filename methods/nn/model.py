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
        input_size=longest_seq*embedding_dim
        output_size=longest_seq
        
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
        
        #print(out.shape)
        #print(targets.shape)

        loss = nn.MSELoss()
        loss = loss(out,targets)
        
        
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