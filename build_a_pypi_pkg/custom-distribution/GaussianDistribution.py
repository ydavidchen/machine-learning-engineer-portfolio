# Template code provided by Udacity
# Code passed Udacity's black-box unit tests

import math;
import matplotlib.pyplot as plt;

class Gaussian(Distribution):
    """
    Inherited from generic distribution class; defaults to standard-normal
    :param: mean (float) representing the mean value of the distribution
    :param: stdev (float) representing the standard deviation of the distribution
    """
    def __init__(self, mu=0, sigma=1):
        Distribution.__init__(self, mu, sigma);

    def calculate_mean(self):
        """
        Calculates the mean of the data set.
        :params: None
        :return: float: mean of the data set
        """
        self.mean = sum(self.data) / len(self.data);
        return self.mean;

    def calculate_stdev(self, sample=True):
        """
        Computes either sample or population SD
        :params: sample (bool): whether the data represents a sample or population
        :return: float: standard deviation of the data set
        """
        n = len(self.data);
        denom = (n-1) if sample else n;

        vec = [(x - self.mean)**2 for x in self.data];
        self.stdev = math.sqrt( sum(vec) / denom);
        return self.stdev;

    def plot_histogram(self):
        """
        Uses matplotlib to sketch histogram
        :params: None
        :returns: None
        """
        plt.hist(self.data);
        plt.title("Histogram of Data");
        plt.xlabel("Data");
        plt.ylabel("Count");
        plt.show();

    def pdf(self, x):
        """
        Computes PDF for Gaussian
        :paramsx (float): point for calculating the probability density function
        Returns:
            float: probability density function output
        """
        mu, s = self.mean, self.stdev;
        const = 1 / (s * math.sqrt(2*math.pi));
        z = (x - mu) / s;
        return const * math.exp(-0.5 * z**2);
