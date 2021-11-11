from matplotlib import pyplot as plt

tc = [-38.19, -31.38, -28.52, -26.75, -24.349]
sc = [-44.7, -40.737, -38.95, -37.68, -35.7]
sc_white = [-35.48, -28.26, -25.16, -23.68, -21.240]
iso_list = [100, 400, 800, 1600, 3200]
plt.plot(iso_list, tc, label="Two Colors")
plt.plot(iso_list, sc, label="Single Colors (Dark)")
plt.plot(iso_list, sc_white, label="Single Colors (Bright)")
plt.grid()
plt.ylabel("MSE[dB]")
plt.xlabel("ISO")
plt.legend()
plt.show()
