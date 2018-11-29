from factory.dataFactory import getData
from model.go_model import GoModel


def jm():
    data = getData()

    jm_model = GoModel(data)
    a, b = jm_model.process()
    print('a=' + str(a) + ' b=' + str(b))


if __name__ == '__main__':
    jm()
