import gcrb
import torch
from experiments import constants
import pickle
from pytorch_model.noise_flow import generate_noisy_image_flow
from experiments.data_model.edge_position.edge_image_generator import EdgeImageGenerator
from experiments.analysis.analysis_helpers import image_channel_swipe_nhwc2nchw

if __name__ == '__main__':
    batch_size = 32
    patch_size = 32
    width_array = [1.5, 2, 4, 8, 16, 24, 31]
    # cross_point_array = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 30]
    iso_array = [100]
    color_swip = False
    cam_array = [0, 2]

    flow = generate_noisy_image_flow([4, patch_size, patch_size], device=constants.DEVICE, load_model=True,
                                     activation_function="silu").to(
        constants.DEVICE)
    eig = EdgeImageGenerator(patch_size)

    results = {}
    for cam in cam_array:
        results_iso = {}
        for iso in iso_array:
            results_edge_width = {}
            for edge_width in width_array:
                generate_image = eig.get_image_function(edge_width, color_swip)


                def sample_function(in_batch_size, in_theta):
                    status = torch.ones([in_batch_size], device=constants.DEVICE).bool()
                    bayer_img = generate_image(in_theta)
                    in_cond_vector = [image_channel_swipe_nhwc2nchw(bayer_img),
                                      torch.tensor(iso, device=constants.DEVICE),
                                      torch.tensor(cam, device=constants.DEVICE)]
                    return flow.sample_nll(in_batch_size, cond=in_cond_vector).reshape([-1, 1]), status


                _results_croos_points = []
                for cross_point in constants.CROSS_POINT:
                    theta_vector = cross_point * torch.ones(batch_size, requires_grad=True).to(constants.DEVICE)
                    gfim = gcrb.sampling_gfim(sample_function, theta_vector.reshape([-1, 1]),
                                              batch_size=batch_size,
                                              m=64000)
                    psnr = torch.linalg.inv(gfim).diagonal().mean()
                    _results_croos_points.append(psnr.item())
                    print(psnr)

                results_edge_width[edge_width] = _results_croos_points
            results_iso[iso] = results_edge_width
        results[cam] = results_iso

    file_name = "new_results_edge" if not color_swip else "new_results_edge_swip"
    with open(f"{file_name}.pickle", "wb") as file:
        pickle.dump(results, file)
