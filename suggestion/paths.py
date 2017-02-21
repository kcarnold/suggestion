import os


class paths:
    parent = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    ui = os.path.join(parent, 'frontend/build')
    logdir = os.path.join(parent, 'logs')
    db = os.path.join(parent, 'database.db')

    cache = os.path.join(parent, 'cache')
    models = os.path.join(parent, 'models')

    @classmethod
    def model_basename(cls, model_name):
        return os.path.join(cls.models, model_name)
