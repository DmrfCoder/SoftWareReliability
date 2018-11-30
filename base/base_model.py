from abc import ABCMeta, abstractmethod


class baseModel(metaclass=ABCMeta):

    def __init__(self, parameters):
        self.parameters = parameters
        self.data = parameters['resource']
        self.data_count = self.data.shape[0] - 1
        self.data_index_len = self.data.shape[0]

    @abstractmethod
    def process(self):
        raise NotImplementedError('process method is not implemented')
