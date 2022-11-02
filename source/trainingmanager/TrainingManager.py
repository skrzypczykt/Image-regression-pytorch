import torch
import tqdm as tqdm

from source.utils import denormalize_year


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


# training function
def fit(model, train_dataloader, train_dataset, optimizer, criterion):
    print('Training')
    model.train()
    train_running_loss = 0.0
    train_accuracy = 0
    counter = 0
    total = 0
    prog_bar = tqdm.tqdm(enumerate(train_dataloader), total=int(len(train_dataset) / train_dataloader.batch_size))
    for i, data in prog_bar:
        counter += 1
        data, target = data[0], data[1]
        year = denormalize_year(target)
        total += target.size(0)
        optimizer.zero_grad()
        outputs = model(data)
        denormalized_outputs = denormalize_year(outputs)
        loss = criterion(outputs, target)
        train_running_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_accuracy = (year == denormalized_outputs).sum().item()

    train_loss = train_running_loss / counter
    train_accuracy = train_accuracy / counter
    return train_loss, train_accuracy


# validation function
def validate(model, test_dataloader, val_dataset, criterion):
    print('Validating')
    model.eval()
    val_running_loss = 0.0
    counter = 0
    total = 0
    prog_bar = tqdm.tqdm(enumerate(test_dataloader), total=int(len(val_dataset) / test_dataloader.batch_size))
    with torch.no_grad():
        for i, data in prog_bar:
            counter += 1
            data, target = data[0], data[1]
            total += target.size(0)
            outputs = model(data)
            loss = criterion(outputs, target)

            val_running_loss += loss.item()

        val_loss = val_running_loss / counter

        return val_loss
