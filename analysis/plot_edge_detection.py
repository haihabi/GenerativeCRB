from matplotlib import pyplot as plt

width_array = [1, 1.1, 1.5, 2, 4, 6, 8, 16, 24, 31]
width_results = [-41.4398, -47.0369, -45.4288, -44.9259, -44.4628, -43.4858, -42.6747, -40.2980, -37.9747, -36.4974]

plt.plot(width_array, width_results)
plt.xlabel("Edge width")
plt.ylabel("MSE[dB]")
plt.grid()
plt.show()

position_results = [-45.1646, -45.1061, -45.0113, -45.1221, -44.9504, -44.9909, -44.9866]
position_results_cross = [-45.0214, -49.5257, -48.2796, -48.3650, -48.3871, -48.3560, -48.3051]
cross_point_array = [1, 2, 4, 8, 16, 24, 31]

position_results = [ -45.0113, -45.1221, -44.9504, -44.9909, -44.9866]
position_results_cross = [ -48.2796, -48.3650, -48.3871, -48.3560, -48.3051]
cross_point_array = [ 4, 8, 16, 24, 31]
plt.subplot(1,2,1)
plt.plot(cross_point_array,position_results)
plt.xlabel("Edge position")
plt.ylabel("MSE[dB]")
plt.grid()
plt.title("Original Colors")
plt.subplot(1,2,2)
plt.plot(cross_point_array,position_results_cross)
plt.xlabel("Edge position")
plt.ylabel("MSE[dB]")
plt.title("Swap colors")
plt.grid()
plt.show()
