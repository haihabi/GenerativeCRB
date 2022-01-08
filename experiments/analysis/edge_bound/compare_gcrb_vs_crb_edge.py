import numpy as np
import pickle
import gcrb
from experiments import constants
import torch
from training_nlf.training_main import generate_nlf_flow
from experiments.analysis.edge_bound.edge_image_generator import EdgeImageGenerator
from matplotlib import pyplot as plt
from experiments.analysis.edge_bound.nlf_crb import get_crb_function


def image2vector(in_image):
    return torch.permute(in_image, (0, 3, 1, 2))


def load_nlf_flow(in_input_shape, in_generate_image, model_parameter_path, in_iso, in_cam):
    flow = generate_nlf_flow(in_input_shape, True, noise_only=False)
    parameter_nlf = torch.load(model_parameter_path,
                               map_location="cpu")
    flow.flow.flows[1].alpha.data = parameter_nlf["flow.flows.0.alpha"]
    flow.flow.flows[1].delta.data = parameter_nlf["flow.flows.0.delta"]
    print(flow.flow.flows[1].alpha.data)
    print(flow.flow.flows[1].delta.data)
    flow = flow.to(constants.DEVICE)

    def sample_function(in_batch_size, in_theta):
        bayer_img = in_generate_image(in_theta)
        in_cond_vector = [image2vector(bayer_img),
                          torch.tensor(in_iso, device=constants.DEVICE).long().reshape([-1]).repeat([in_batch_size]),
                          torch.tensor(in_cam, device=constants.DEVICE).long().reshape([-1]).repeat([in_batch_size])]
        return flow.sample_nll(in_batch_size, cond=in_cond_vector).reshape([-1, 1])

    iso_index = constants.ISO_LIST.index(in_iso)
    alpha = flow.flow.flows[1].alpha.data.detach().cpu().numpy()[iso_index, cam]
    delta = flow.flow.flows[1].delta.data.detach().cpu().numpy()[iso_index, cam]
    return sample_function, alpha, delta


def loop_gcrb_cross_point(in_batch_size, in_sample_function, in_cross_point_array):
    results_croos_points_gaussian = []
    for cross_point in in_cross_point_array:
        theta_vector = cross_point * torch.ones(in_batch_size, requires_grad=True).to(constants.DEVICE)
        gfim = gcrb.sampling_gfim(in_sample_function, theta_vector.reshape([-1, 1]),
                                  batch_size=in_batch_size,
                                  m=64000)
        results_croos_points_gaussian.append(1 / gfim.cpu().detach().numpy().flatten())
    return 10 * np.log10(np.asarray(results_croos_points_gaussian).flatten())


def loop_crb_cross_point(in_crb_function, in_cross_point_array):
    return 10 * np.log10(np.asarray([in_crb_function(cross_point) for cross_point in in_cross_point_array]))


if __name__ == '__main__':
    cam = 2
    edge_width = 8
    iso = 100
    patch_size = 32
    color_swip = False

    model_path_gaussian = "training_nlf/flow_gaussian_best.pt"
    model_path_nlf = "training_nlf/flow_nlf_best.pt"
    cross_point_array = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 30]
    input_shape = [4, patch_size, patch_size]

    with open("results_edge.pickle", "rb") as file:
        data = pickle.load(file)

    print(f"Comparing CRB using Device:{constants.INDEX2CAM[cam]} with ISO {iso} and Edge Width:{edge_width}")

    gcrb_db = data[cam][iso][edge_width]
    eig = EdgeImageGenerator(patch_size)
    generate_image = eig.get_image_function(edge_width, color_swip)

    flow_gaussian_sample_func, _, sigma = load_nlf_flow(input_shape, generate_image, model_path_gaussian, iso, cam)
    crb_function = get_crb_function(0, sigma, edge_width, color_swip, eig)
    results_gaussian_crb = loop_crb_cross_point(crb_function, cross_point_array)

    flow_nlf_sample_func, alpha, delta = load_nlf_flow(input_shape, generate_image, model_path_nlf, iso, cam)
    crb_function = get_crb_function(alpha, delta, edge_width, color_swip, eig)
    results_nlf_crb = loop_crb_cross_point(crb_function, cross_point_array)

    plt.plot(cross_point_array, results_gaussian_crb, label="Gaussian")
    plt.plot(cross_point_array, results_nlf_crb, label="NLF")
    plt.plot(cross_point_array, np.asarray(gcrb_db).flatten(), label="GCRB")
    plt.grid()
    plt.legend()
    plt.xlabel("Edge Position")
    plt.ylabel("MSE[dB]")
    plt.show()
