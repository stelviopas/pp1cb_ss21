import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from methods.nn.ffnet import FFNet

def nested_cross_validation(dataset,
                            hparams,
                            mode='print_fold_info',
                            k=10, max_epochs=1, *kwargs):
    # Set fixed random number seed
    SEED = 1
    torch.manual_seed(SEED)

    # Define the K-fold Cross Validator
    skf = StratifiedKFold(n_splits=k, random_state=SEED, shuffle=True)

    # For folds results
    fold_errors = {}

    fold_test_predictions = {}

    # For debugging loaders
    loaders = {}

    # Nested K-Fold Cross Validation model evaluation
    # We split the data stratified on artificially constructed bins in df['bins'
    # and extract indices.

    # By splitting we only extract indices of samples for each test/train/val sets,
    # thus we only need either X or y (equal length).
    # For stratification, however, we require the artificially assigned bins which are also defined in the
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
            batch_size=hparams["batch_size"],
            sampler=val_ids)

        trainloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=hparams["batch_size"], sampler=train_ids)

        testloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=hparams["batch_size"], sampler=test_ids)

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

          model=FFNet(hparams=hparams)

          early_stopping = EarlyStopping('val_loss')

          trainer = pl.Trainer(weights_summary=None,
                                max_epochs=max_epochs,
                                deterministic=True,
                                callbacks=[early_stopping],
                                gpus=1,
                                auto_select_gpus=1
                                #auto_lr_find=True, #TO DO
                                #auto_scale_batch_size=True, # TO DO
                                )
          trainer.fit(model, train_dataloader=trainloader, val_dataloaders=valloader)
          trainer.test(model, test_dataloaders=testloader)
          test_loss = model.test_results
          fold_errors[fold] = test_loss
          test_prediction = model.truth_prediction_test
          fold_test_predictions[fold] = test_prediction
          
        else:
            print("Skip!")
            

    if mode == 'evaluate':
        return fold_errors, fold_test_predictions
    else:
        return loaders
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from methods.nn.ffnet import FFNet

def nested_cross_validation(dataset,
                            hparams,
                            mode='print_fold_info',
                            k=10, max_epochs=1, *kwargs):
    # Set fixed random number seed
    SEED = 1
    torch.manual_seed(SEED)

    # Define the K-fold Cross Validator
    skf = StratifiedKFold(n_splits=k, random_state=SEED, shuffle=True)

    # For folds results
    fold_errors = {}

    fold_test_predictions = {}

    # For debugging loaders
    loaders = {}

    # Nested K-Fold Cross Validation model evaluation
    # We split the data stratified on artificially constructed bins in df['bins'
    # and extract indices.

    # By splitting we only extract indices of samples for each test/train/val sets,
    # thus we only need either X or y (equal length).
    # For stratification, however, we require the artificially assigned bins which are also defined in the
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
            batch_size=hparams["batch_size"],
            sampler=val_ids)

        trainloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=hparams["batch_size"], sampler=train_ids)

        testloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=hparams["batch_size"], sampler=test_ids)

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

          model=FFNet(hparams=hparams)

          early_stopping = EarlyStopping('val_loss')

          trainer = pl.Trainer(weights_summary=None,
                                max_epochs=max_epochs,
                                deterministic=True,
                                callbacks=[early_stopping],
                                gpus=1,
                                auto_select_gpus=1
                                #auto_lr_find=True, #TO DO
                                #auto_scale_batch_size=True, # TO DO
                                )
          trainer.fit(model, train_dataloader=trainloader, val_dataloaders=valloader)
          trainer.test(model, test_dataloaders=testloader)
          test_loss = model.test_results
          fold_errors[fold] = test_loss
          test_prediction = model.truth_prediction_test
          fold_test_predictions[fold] = test_prediction
          break
          
        else:
            print("Skip!")
            

    if mode == 'evaluate':
        return fold_errors, fold_test_predictions
    else:
        return loaders
