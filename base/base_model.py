from abc import ABCMeta, abstractmethod


class baseModel(metaclass=ABCMeta):

    def __init__(self, parameters):
        self.parameters = parameters
        self.data = parameters['data']

    @abstractmethod
    def process(self):
        raise NotImplementedError('process method is not implemented')
