import numpy as np
from experiments import constants
import torch
from experiments.analysis.analysis_helpers import image_channel_swipe_nhwc2nchw, image_shape
from pytorch_model.noise_flow import generate_noisy_image_flow
from experiments.analysis.edge_bound.edge_image_generator import EdgeImageGenerator
from torch import nn
from tqdm import tqdm


def conv_bn_nl(in_channels, out_channels, kernel_size, stride, non_linear):
    padding = int((kernel_size - 1) / 2)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), stride=(stride, stride),
                  padding=padding,
                  bias=False),
        nn.BatchNorm2d(out_channels), non_linear())


class ResidualBlock(nn.Module):
    def __init__(self, n_input, n_hidden, non_linear):
        super().__init__()
        self.conv1 = conv_bn_nl(n_input, n_hidden, 3, 1, non_linear)
        self.conv2 = conv_bn_nl(n_hidden, n_input, 3, 1, nn.Identity)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        return x + x1


class MVUENet(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        self.stem = conv_bn_nl(4, n_hidden, 7, 2, nn.ReLU)
        self.block1 = ResidualBlock(n_hidden, 64, nn.ReLU)
        self.predictor = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(n_hidden, 1, bias=False))

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)

        return self.predictor(x)


if __name__ == '__main__':
    iso = 100
    cam = 2
    edge_width = 8
    patch_size = 32
    batch_size = 128
    n_epochs = 5
    n_iter = 1000
    lambda_bias = 10
    color_swip = False
    flow = generate_noisy_image_flow(image_shape(patch_size), device=constants.DEVICE, load_model=True).to(
        constants.DEVICE)
    eig = EdgeImageGenerator(patch_size)
    p_h, p_l = eig.get_pixel_color(color_swip=color_swip)
    generate_image = eig.get_image_function(edge_width, color_swip)
    mvue = MVUENet(128).to(constants.DEVICE)


    def sample_function(in_batch_size, in_theta):
        bayer_img = generate_image(in_theta)
        in_cond_vector = [image_channel_swipe_nhwc2nchw(bayer_img), iso, cam]
        return flow.sample(in_batch_size, in_cond_vector)[-1]


    start_lambda = 0.001
    end_lambda = 10
    alpha = (end_lambda - start_lambda) / 5

    opt = torch.optim.SGD(mvue.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4)
    for i in range(n_epochs):
        with tqdm(total=n_iter) as pbar:
            mse_list = []
            bias_list = []
            lambda_bias = alpha * i + start_lambda
            for _ in range(n_iter):
                theta = np.random.choice(constants.CROSS_POINT, size=batch_size)
                theta = torch.tensor(theta).to(constants.DEVICE)
                data = sample_function(batch_size, theta)
                theta_hat = mvue(data)
                mse = torch.pow(theta_hat.flatten() - theta, 2.0).mean()
                bias = 0
                for c in constants.CROSS_POINT:
                    theta = c * torch.ones(batch_size).to(constants.DEVICE)
                    data = sample_function(batch_size, theta)
                    theta_hat = mvue(data)
                    bias += torch.pow(theta_hat.mean() - theta.mean(), 2.0)
                bias = bias / len(constants.CROSS_POINT)
                loss = mse + lambda_bias * bias
                opt.zero_grad()
                loss.backward()
                opt.step()
                mse_list.append(mse.item())
                bias_list.append(bias.item())
                pbar.update(1)
                pbar.set_postfix({'mse': np.mean(mse_list), 'bias': np.mean(bias_list)})
        pass
    torch.save(mvue.state_dict(), "mvue.pt")
