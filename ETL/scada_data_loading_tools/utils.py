import os


def iter_dirs(path):
    'Iterate over directories in a given path.'
    yield from (f.name for f in os.scandir(path) if f.is_dir())


def set_relative_index(data):
    'Return a copy of a pandas Series or Dataframe with relative index.'
    data = data.copy()
    data.index -= data.index[0]
    return data
