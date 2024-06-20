import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load the estimated times
    estimated_times = np.load("data/sample_estimate_times.npy")
    estimated_times = estimated_times[estimated_times < 10000]
    estimated_times = estimated_times[estimated_times > 0]

    # Calculate the mean and standard deviation
    mean = np.mean(estimated_times)
    std = np.std(estimated_times)
    print(f"Mean: {mean}")
    print(f"Standard deviation: {std}")
    plt.hist(estimated_times, bins=100, density=True)
    plt.title("Estimate time")
    plt.savefig("interactive/estimated_times_histogram.png")
