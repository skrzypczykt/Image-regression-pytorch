from datetime import datetime
from ax.service.managed_loop import optimize
import matplotlib.pyplot as plt
import numpy as np
import torch.nn
from torch.utils.data import DataLoader
from torchsummary import summary

from source.data_processing.dataset import WearingDataset

# data_processing
from source.models.resnet18 import ResnetRegression
from source.trainingmanager.TrainingManager import EarlyStopping, fit, validate

TRAINING_DATA = "../data/dresses_train.csv"

VAL_DATA = "../data/dresses_val.csv"
# # model
# MODEL_PATH = f"../results/ResNet/model_transfer_learning_{datetime.datetime.now().date()}"


if __name__ == '__main__':
    """# Data loading and preprocessing"""
    batch_size = 32

    train_ds = WearingDataset(TRAINING_DATA, transform=True)
    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=2)

    val_ds = WearingDataset(VAL_DATA, transform=True)
    val_loader = DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    PATH = "model_{model_name}_epoch_{epoch}_{date}"


    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    # summary(model, input_size=(3, 224, 224))

    def train_evaluate(parameterization):

        model = ResnetRegression(n_fc_layers=parameterization.get("n_fc_layers", 1),
                                 n_neurons=parameterization.get("n_neurons", 64))

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        criterion = torch.nn.MSELoss()
        metric = torch.nn.L1Loss()

        # instantiate Early stopping object
        early_stopping = EarlyStopping()

        # loss plot names and model name
        loss_plot_name = 'es_loss'
        model_name = 'es_model'

        # num of epochs
        epochs = 20

        # losses initialization
        train_loss = []
        val_loss = []

        # best loss initialization
        best_val_loss = 10000

        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch} of {epochs}")
            train_epoch_loss, train_accuracy = fit(
                model, train_loader, train_ds, optimizer, criterion
            )
            val_epoch_loss = validate(
                model, val_loader, val_ds, criterion
            )
            train_loss.append(train_epoch_loss)
            val_loss.append(val_epoch_loss)
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss

            print(f"Train Loss: {train_epoch_loss:.4f}")
            print(f"Train Accuracy: {train_accuracy:.4f}")
            print(f'Val Loss: {val_epoch_loss:.4f}')

            # saving best model only
            # if epoch == 1 or best_val_loss == val_epoch_loss:
            #     torch.save({
            #         'epoch': epoch,
            #         'model_state_dict': model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'loss': val_epoch_loss,
            #     }, PATH.format(model_name=model_name, epoch=epoch, date=datetime.now().date))

            # early stopping
            early_stopping(val_epoch_loss)
            if early_stopping.early_stop:
                break

        print('Finished Training')
        return {"loss": best_val_loss}


    best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "n_fc_layers", "type": "range", "bounds": [1, 4]},
            {"name": "n_neurons", "type": "range", "bounds": [16, 128]},
            # {"name": "momentum", "type": "range", "bounds": [0.0, 1.0]},
            # {"name": "max_epoch", "type": "range", "bounds": [1, 30]},
            # {"name": "stepsize", "type": "range", "bounds": [20, 40]},
        ],

        evaluation_function=train_evaluate,
        objective_name='loss',
        minimize=True,
        total_trials=5
    )

    print(best_parameters)
    means, covariances = values
    print(means)
    print(covariances)
