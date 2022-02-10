import numpy as np
from experiments import constants
import torch
from experiments.analysis.analysis_helpers import image_channel_swipe_nhwc2nchw, image_shape
from pytorch_model.noise_flow import generate_noisy_image_flow
from experiments.data_model.edge_position.edge_image_generator import EdgeImageGenerator
from torch import nn
from tqdm import tqdm
from experiments.analysis.analysis_helpers import db
import math
from experiments.common.numpy_dataset import NumpyDataset


def conv_bn_nl(in_channels, out_channels, kernel_size, stride, non_linear):
    padding = int((kernel_size - 1) / 2)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), stride=(stride, stride),
                  padding=padding,
                  bias=False),
        nn.BatchNorm2d(out_channels), non_linear())


class GlobalPool(nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.mean(x, dim=(2, 3), keepdim=True)


class AttentionBlock(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        r = int(n_channels / 4) + 1
        self.se_attention = nn.Sequential(GlobalPool(),
                                          nn.Conv2d(n_channels, r, kernel_size=1, stride=1, padding=0, bias=True),
                                          nn.GELU(),
                                          nn.Conv2d(r, n_channels, kernel_size=1, stride=1, padding=0, bias=True),
                                          nn.Sigmoid())

    def forward(self, x):
        ch_att = self.se_attention(x)
        return x * ch_att


class ResidualBlock(nn.Module):
    def __init__(self, n_input, n_hidden, non_linear):
        super().__init__()
        self.conv1 = conv_bn_nl(n_input, n_hidden, 3, 1, non_linear)
        self.conv2 = conv_bn_nl(n_hidden, n_input, 3, 1, nn.Identity)
        self.ab = AttentionBlock(n_input)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        return self.ab(x + x1)


class MVUENet(nn.Module):
    def __init__(self, n_hidden, p_h, p_l, num_heads=4):
        super().__init__()
        self.p_h = nn.Parameter(p_h.reshape([1, -1, 1, 1]), requires_grad=False)
        self.p_l = nn.Parameter(p_l.reshape([1, -1, 1, 1]), requires_grad=False)
        # self.ln_input = nn.LayerNorm([4, 1, 1], elementwise_affine=False)
        self.stem = conv_bn_nl(4, n_hidden, 3, 1, nn.GELU)
        self.stem1 = conv_bn_nl(n_hidden, n_hidden, 3, 1, nn.GELU)
        self.ln1 = nn.LayerNorm(n_hidden)
        self.att1 = nn.MultiheadAttention(n_hidden, num_heads=num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(n_hidden)
        self.att2 = nn.MultiheadAttention(n_hidden, num_heads=num_heads, batch_first=True)
        # self.block1 = nn.Sequential(ResidualBlock(n_hidden, 64, nn.GELU), ResidualBlock(n_hidden, 64, nn.GELU))
        # self.mid_conv = conv_bn_nl(n_hidden, 2 * n_hidden, 3, 1, nn.GELU)
        # self.block2 = nn.Sequential(ResidualBlock(2 * n_hidden, 128, nn.GELU),
        #                             ResidualBlock(2 * n_hidden, 128, nn.GELU))
        # self.predictor = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(2 * n_hidden, 1, bias=False))
        self.predictor = nn.Linear(n_hidden, 1, bias=True)

    def forward(self, x):
        x = (x - self.p_h) / (self.p_h - self.p_l)
        # x = self.ln_input(x)
        x = self.stem(x)
        x = self.stem1(x)
        x = x.mean(dim=-1)
        x = x.transpose(dim0=1, dim1=2)
        x = self.ln1(x)
        x, att1 = self.att1(x, x, x)
        x = self.ln2(x)
        x, att2 = self.att2(x, x, x)
        x = x.mean(dim=1)
        # x = self.block1(x)
        # x = self.mid_conv(x)
        # x = self.block2(x)

        return self.predictor(x)


if __name__ == '__main__':
    iso = 800
    cam = 2
    edge_width = 2
    patch_size = 32
    batch_size = 128
    n_epochs = 50

    lambda_bias = 10
    dataset_size = 50000
    n_iter = math.ceil(dataset_size / batch_size)
    color_swip = False
    flow = generate_noisy_image_flow(image_shape(patch_size), device=constants.DEVICE, load_model=True).to(
        constants.DEVICE)
    eig = EdgeImageGenerator(patch_size)
    p_h, p_l = eig.get_pixel_color(color_swip=color_swip)
    generate_image = eig.get_image_function(edge_width, color_swip)
    mvue = MVUENet(128, p_h, p_l).to(constants.DEVICE).train()


    def sample_function(in_batch_size, in_theta):
        bayer_img = generate_image(in_theta)
        in_cond_vector = [image_channel_swipe_nhwc2nchw(bayer_img), iso, cam]
        return flow.sample(in_batch_size, in_cond_vector, temperature=0.6)


    start_lambda = 0.001
    end_lambda = 10
    alpha = (end_lambda - start_lambda) / 5

    opt = torch.optim.AdamW(mvue.parameters(), lr=0.001, weight_decay=1e-4)
    lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, [30, 40])
    theta_min = min(constants.CROSS_POINT)
    theta_max = max(constants.CROSS_POINT)

    for i in range(n_epochs):
        with tqdm(total=n_iter) as pbar:
            mse_list = []
            bias_list = []
            lambda_bias = alpha * i + start_lambda
            if i == 0:
                data_list = []
                parameters_list = []
                for _ in range(n_iter):
                    theta = theta_min + (theta_max - theta_min) * torch.rand(batch_size).to(constants.DEVICE)
                    data = sample_function(batch_size, theta)
                    data_list.append(data.detach().cpu().numpy())
                    parameters_list.append(theta.detach().cpu().numpy())

                    theta_hat = mvue(data)
                    mse = torch.pow(theta_hat.flatten() - theta, 2.0).mean()

                    loss = mse
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    mse_list.append(mse.item())
                    pbar.update(1)
                    pbar.set_postfix(
                        {'mse': np.mean(mse_list), 'bias': np.mean(bias_list), "epoch": i,
                         'mse_db': db(np.mean(mse_list))})
                dataset = NumpyDataset(np.concatenate(data_list), np.concatenate(parameters_list))
                training_dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                                      shuffle=True, num_workers=4)
            else:
                for data, theta in training_dataset_loader:
                    data = data.to(constants.DEVICE)
                    theta = theta.to(constants.DEVICE)
                    theta_hat = mvue(data)
                    mse = torch.pow(theta_hat.flatten() - theta, 2.0).mean()

                    loss = mse
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    mse_list.append(mse.item())
                    pbar.update(1)
                    pbar.set_postfix(
                        {'mse': np.mean(mse_list), 'bias': np.mean(bias_list), "epoch": i,
                         'mse_db': db(np.mean(mse_list))})

            lr_decay.step(i)
        torch.save(mvue.state_dict(), "mvue.pt")
