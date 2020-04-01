# Template code provided by Udacity
# Code passed Udacity's black-box unit tests

class Distribution:
    def __init__(self, mu=0, sigma=1):
        """
        :param: mean (float) representing the mean value of the distribution
        :param: stdev (float) representing the standard deviation of the distribution
        :param: data_list (list of floats) a list of floats extracted from the data file
        """
        self.mean = mu;
        self.stdev = sigma;
        self.data = [];

    def read_data_from_file(self, file_name):
        """
        :param: file_name (str): Name of a file to read from
        :returns: None
        """
        with open(file_name) as file:
            data_list = [];
            line = file.readline();
            while line:
                data_list.append(int(line));
                line = file.readline();
        file.close();
        self.data = data_list;
