from abc import ABCMeta, abstractmethod


class baseModel(metaclass=ABCMeta):

    def __init__(self, data):
        self.data = data

    @abstractmethod
    def process(self):
        raise NotImplementedError('process method is not implemented')
