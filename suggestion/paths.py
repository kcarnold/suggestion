import pathlib


class paths:
    parent = pathlib.Path(__file__).resolve().parent.parent
    ui = parent / 'frontend/build'
    logdir = parent / 'logs'
    db = parent / 'database.db'
    preproc = parent / 'yelp_preproc'
    data = parent / 'data'

    cache = parent / 'cache'
    models = parent / 'models'

    @classmethod
    def model_basename(cls, model_name):
        return cls.models / model_name
