import numpy as np
import time
from pydmd.bopdmd import BOPDMD


def simulate_data(dim, num_samples):
    """
    Function to simulate a data matrix of shape `(dim, num_samples)`.
    """
    return np.random.normal(0, 1, size=(dim, num_samples))


def time_bopdmd(dim, num_samples):
    """
    Function to benchmark the BOPDMD with increasing dimension and number of samples.
    The function times how long it takes to fit the data and return the time.
    """
    X = simulate_data(dim, num_samples)
    t = np.arange(num_samples)

    bopdmd = BOPDMD(svd_rank=min(dim, num_samples))

    start_time = time.time()
    bopdmd.fit(X, t)
    total_time = time.time() - start_time

    return total_time


if __name__ == "__main__":
    for dim in [10, 10]:
        for num_samples in [
            1000,
            5000,
        ]:

            time_taken = time_bopdmd(dim, num_samples)
            print(
                f"Time taken for BOPDMD with dimension {dim} and samples {num_samples}: {time_taken}"
            )
