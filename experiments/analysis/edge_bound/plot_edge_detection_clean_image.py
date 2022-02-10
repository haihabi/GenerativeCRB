import torch
from experiments import constants

# from sidd.pipeline import process_sidd_image
from matplotlib import pyplot as plt
from experiments.data_model.edge_position.edge_image_generator import EdgeImageGenerator
from experiments.analysis.analysis_helpers import rggb2rgb

# def unpack_raw(raw4ch):
#     """Unpacks 4 channels to Bayer image (h/2, w/2, 4) --> (h, w)."""
#     img_shape = raw4ch.shape
#     h = img_shape[0]
#     w = img_shape[1]
#     # d = img_shape[2]
#     bayer = np.zeros([h * 2, w * 2], dtype=np.float32)
#     # bayer = raw4ch
#     # bayer.reshape((h * 2, w * 2))
#     bayer[0::2, 0::2] = raw4ch[:, :, 0]
#     bayer[0::2, 1::2] = raw4ch[:, :, 1]
#     bayer[1::2, 1::2] = raw4ch[:, :, 2]
#     bayer[1::2, 0::2] = raw4ch[:, :, 3]
#     return bayer
#
#
# with open("../metadata_edge.pickle", "rb") as f:
#     bayer_2by2, wb, cst2 = pickle.load(f)

batch_size = 1
gef = EdgeImageGenerator(patch_size=32)
generate_image = gef.get_image_function(2, color_swip=False)
cross_point = 16

theta_vector = cross_point * torch.ones(batch_size, requires_grad=True).to(constants.DEVICE)
I = generate_image(theta_vector)
I = I.cpu().detach().numpy()[0, :, :, :]
# I = process_sidd_image(unpack_raw(I), bayer_2by2, wb, cst2)
plt.subplot(2, 2, 1)
plt.imshow(rggb2rgb(I))
plt.axis('off')
plt.title(f"Position:{cross_point}, Edge Width:{2}")
plt.subplot(2, 2, 2)
cross_point = 2
theta_vector = cross_point * torch.ones(batch_size, requires_grad=True).to(constants.DEVICE)
I = generate_image(theta_vector)
I = I.cpu().detach().numpy()[0, :, :, :]
# I = process_sidd_image(unpack_raw(I), bayer_2by2, wb, cst2)
plt.imshow(rggb2rgb(I))
plt.axis('off')
plt.title(f"Position:{cross_point}, Edge Width:{2}")
plt.subplot(2, 2, 3)
cross_point = 16
theta_vector = cross_point * torch.ones(batch_size, requires_grad=True).to(constants.DEVICE)
generate_image = gef.get_image_function(4, color_swip=False)
I = generate_image(theta_vector)
I = I.cpu().detach().numpy()[0, :, :, :]
# I = process_sidd_image(unpack_raw(I), bayer_2by2, wb, cst2)
plt.imshow(rggb2rgb(I))
plt.axis('off')
plt.title(f"Position:{cross_point}, Edge Width:{4}")
plt.subplot(2, 2, 4)
cross_point = 16
theta_vector = cross_point * torch.ones(batch_size, requires_grad=True).to(constants.DEVICE)
generate_image = gef.get_image_function(1, color_swip=False)
I = generate_image(theta_vector)
I = I.cpu().detach().numpy()[0, :, :, :]
plt.imshow(rggb2rgb(I))
plt.axis('off')
plt.title(f"Position:{cross_point}, Edge Width:{1}")
plt.show()
