from factory.graph_data_factory import load_dataset
from model.go_model import GoModel
from model.jm_model import JmModel


def go():
    data = load_dataset()

    go_parmars = {
        'resource': data,
        'eir': 0.1
    }

    go_model = GoModel(parameters=go_parmars)
    a, b = go_model.process()
    print('a=' + str(a) + ' b=' + str(b))


def jm():
    data = load_dataset()
    ev=0.1#input('ev:')
    ex=0.1#input('ex:')

    jm_parmars = {
        'resource': data,
        'ev': float(ev),
        'ex': float(ex)
    }

    jm_model = JmModel(jm_parmars)
    n,fai=jm_model.process()
    print('n=' + str(n) + ' fai=' + str(fai))


if __name__ == '__main__':
    jm()
