import methods.nn.disorder_dataset as dd
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from methods.nn.disorder_dataset import collate
import torch.nn.functional as F


# TODO: outsource this to its own class
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
            batch_size=10,
            collate_fn=collate,
            sampler=val_ids)

        trainloader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=collate,
            batch_size=10, sampler=train_ids)

        testloader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=collate,
            batch_size=10, sampler=test_ids)

        if mode == 'print_fold_info':

            loaders[fold] = [trainloader, valloader, testloader]

            # print('train -  {}, avg_len - {:.2f}  |  val -  {}, avg_len - {:.2f}'.format(
            #     np.sum(np.bincount(stratify_on[train_ids])),
            #     np.sum(dataset.aa_len[train_ids]) / len(train_ids),
            #     np.sum(np.bincount(stratify_on[val_ids])),
            #     np.sum(dataset.aa_len[val_ids]) / len(val_ids)))
            #
            # print('test -  {}, avg_len - {:.2f}'.format(
            #     np.sum(np.bincount(stratify_on[test_ids])),
            #     np.sum(dataset.aa_len[test_ids]) / len(test_ids)))

            show_batches = 5
            for i, batch in enumerate(trainloader):
                # print('Embedding: \n', batch['embeddings'])
                # print('\nZ-scores: \n', batch['z_scores'])
                # print('\nSequence Lengths: \n', batch['lengths'])
                # print('\n')
                if i == show_batches:
                    break

            print()

        elif mode == 'evaluate':
            pass
        else:
            print("Mode is not specified!")
            break

    if mode == 'evaluate':
        return results
    else:
        return loaders


# Define model
# THIS IS A DUMMY MODEL AND NOT ANYTHING LIKE WHAT WILL BE USED LATER
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(1024*10, 1024*10),
            torch.nn.ReLU(),
            torch.nn.Linear(1024*10, 1024*10),
            torch.nn.ReLU(),
            torch.nn.Linear(1024*10, 1024*10),
            torch.nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        print(f"Size of x in the forward method: {x.size()}")
        logits = self.linear_relu_stack(x)
        return logits


if __name__ == "__main__":
    dataset = dd.load_dataset(path="../../data")
    loaders = nested_cross_validation(dataset)

    # for now, we only use the very first fold, but later this should of course be substituted by cv
    fold_1 = loaders[0]
    train = fold_1[0]
    val = fold_1[1]
    test = fold_1[2]

    # in pytorch tutorials, this is how they inspect the data
    # for X, y in test:
    #     print("Shape of X [N, C, H, W]: ", X.shape)
    #     print("Shape of y: ", y.shape, y.dtype)
    #     break
    # TODO: why do we get a dictionary from the dataloader instead of a tuple of x and y?

    # printing an instance of the test set
    instance = next(iter(test))
    print(f"Feature batch shape: {instance['embeddings'].size()}")
    print(f"Labels batch shape: {instance['z_scores'].size()}")
    print(instance['embeddings'][0])
    print(instance['z_scores'][0])
    print(instance['lengths'][0])
    # outputs for shapes:
    # Feature batch shape: torch.Size([145, 10, 1024])   (length?, batch_size?, embedding_size)
    # Labels batch shape: torch.Size([145, 10])          (length?, batch_size?)

    # X = instance['embeddings'][0]
    # y = instance['z_scores'][0]
    #
    # # testing creation of a model
    # model = NeuralNetwork()
    # print(model)
    #
    # loss_fn = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    #
    # # all of the stuff below would usually be executed in a training loop
    # # this is just a simple test
    # size = len(test.dataset)
    # print(f"Size of the dataloaders dataset: {size}")
    #
    # # Compute prediction error
    # pred = model(X)
    # loss = loss_fn(pred, y)
    #
    # # Backpropagation
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    #
    # # len would usually be multiplied with the batch
    # loss, current = loss.item(), len(X)
    # print(f"Loss: {loss}")
