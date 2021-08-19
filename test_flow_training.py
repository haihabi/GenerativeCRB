from flow.flow import Sequential
from flow.conditioner import MADE
from flow.transformer import DSF
from flow.training import get_device, train, plot_losses, test_nll
from data_model.multiplication_example import MultiplicationModel
from flow.modules import BatchNorm, Affine, Sigmoid, Shuffle
import matplotlib.pyplot as plt  # used for plotting flow samples later
import numpy as np
import torch

dim = 4
train_dataset = MultiplicationModel(dim, theta_min=0.2, theta_max=10).build_dataset(50000)
val_dataset = MultiplicationModel(dim, theta_min=0.2, theta_max=10).build_dataset(10000)
device = get_device()  # cuda if available, cpu otherwise
trainX = np.asarray(train_dataset.data)
valX = np.asarray(val_dataset.data)

trainX = torch.from_numpy(trainX).to(device)
valX = torch.from_numpy(valX).to(device)
dim = trainX.size(1)  # dimension of the flow

# X transformed to base distribution U, like so:
# X -> BatchNorm -> MADE-DSF -> BatchNorm -> MADE-DSF -> U
flow = Sequential(
    BatchNorm(dim=dim),  # we use BatchNorm for training stabilization
    MADE(DSF(dim=dim)),  # combines a MADE conditioner with an Affine transformer
    Shuffle(dim=dim),

    BatchNorm(dim=dim),
    MADE(DSF(dim=dim)),
    Shuffle(dim=dim),
    BatchNorm(dim=dim),
    MADE(DSF(dim=dim)),

    Affine(dim=dim)
).to(device)  # don't forget to send it to device

# Train the flow with the train function, but you can use your own.
# train uses early stopping, that's why we need a validation set.
train_losses, val_losses = train(flow, trainX, valX, batch_size=256)

# Plot training and validation losses
plot_losses(train_losses, val_losses)

# # Compute the test set negative log-likelihood (the loss function used for training)
# print(test_nll(flow, testX))
#
# # Generate some samples from the learned distribution
# with torch.no_grad():
#     sample = flow.sample(1000)
#     sample = sample.cpu().numpy()  # to numpy array
#
# # Assuming dim=2, we can plot them with a scatterplot
# plt.scatter(*trainX.numpy().T, alpha=.1, label='real')
# plt.scatter(*sample.T, alpha=.1, label='fake')
# plt.legend()
