import matplotlib.pyplot as plt
import pickle


filename = "models/final_params"
samples = 100

dists = [1040, 1217, 631, 248, 240, 255, 256, 826, 828, 737]
avg_sps = [4.56, 3.48, 6.57, 7.75, 7.50, 9.44, 8.53, 5.10, 5.45, 4.49]

cum_dists = []
deaths = 0

# read in params from loaded model
with open (filename, 'rb') as fp:
    [deaths, _, _, _, cum_dists, _, _] = pickle.load(fp)


# take the last 100 samples
dist_subset = cum_dists[-samples:]
deaths -= samples
# print(dist_subset)
#
# print ("subset:", len(dist_subset), deaths)

bins = [-200, -100, 0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300]


plt.bar(range(deaths,deaths + 100), dist_subset, facecolor='blue')
plt.xlabel('Death #')
plt.ylabel('Distance')
plt.title(r'Distance per death of last ' + str(samples) + " deaths")

plt.show()
