from factory.dataFactory import getData
from model.go_model import GoModel
from model.jm_model import JmModel


def go():
    data = getData()

    go_parmars = {
        'data': data,
        'eir': 0.1
    }

    go_model = GoModel(parameters=go_parmars)
    a, b = go_model.process()
    print('a=' + str(a) + ' b=' + str(b))


def jm():
    data = getData()
    ev=input('ev:')
    ex=input('ex:')
    jm_parmars = {
        'data': data,
        'ev': float(ev),
        'ex': float(ex)
    }

    jm_model = JmModel(jm_parmars)
    n,fai=jm_model.process()
    print('n=' + str(n) + ' fai=' + str(fai))


if __name__ == '__main__':
    jm()
