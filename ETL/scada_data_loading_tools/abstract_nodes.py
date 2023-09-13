from abc import ABCMeta, abstractmethod
import os

import pandas as pd
from tqdm import tqdm

from .utils import iter_dirs


class Node(metaclass=ABCMeta):
    def __init__(self, name, parent_nodes):
        self.name = name
        self.parent_nodes = parent_nodes

        self.dir_path = os.path.join(*parent_nodes, name)
        self.path_nodes = parent_nodes + (name,)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(name={self.name!r}, ' f'parent_nodes={self.parent_nodes!r})'
        )


class BranchNode(Node, metaclass=ABCMeta):
    def iter_child_names(self, subset=None):
        dirs_iterator = iter_dirs(self.dir_path)
        if subset is not None:
            subset = set(subset)
            isin_subset = lambda x: x in subset
            dirs_iterator = filter(isin_subset, dirs_iterator)
        yield from dirs_iterator

    def __getitem__(self, child_name):
        if not os.path.exists(os.path.join(self.dir_path, child_name)):
            raise KeyError(
                f"{self.__class__.__name__} node '{self.name}' has no child with name '{child_name}'."
            )
        return self.child_class(child_name, self.path_nodes)

    def iter_children(self, subset=None):
        return (
            self.child_class(child_name, self.path_nodes)
            for child_name in self.iter_child_names(subset)
        )

    def __iter__(self):
        return self.iter_children()

    def _pop_subset_from_args(self, args, kwargs):
        args = list(args)
        subset = args.pop(0) if args else kwargs.pop(self.child_class.__name__.lower(), None)
        subset = [subset] if isinstance(subset, str) else subset
        return subset, args, kwargs

    def load(self, *args, show_progress=False, **kwargs):
        load_gen = self._load(*args, **kwargs)
        if show_progress:
            len_gen = self.get_len(*args, **kwargs)
            load_gen = tqdm(load_gen, total=len_gen)
        return load_gen

    def _load(self, *args, **kwargs):
        subset, args, kwargs = self._pop_subset_from_args(args, kwargs)
        for child in self.iter_children(subset):
            if isinstance(child, BranchNode):
                yield from child._load(*args, **kwargs)
            else:
                yield child._load(*args, **kwargs)

    def __len__(self):
        return self.get_len()

    def get_len(self, *args, **kwargs):
        subset, args, kwargs = self._pop_subset_from_args(args, kwargs)
        return sum(child.get_len(*args, **kwargs) for child in self.iter_children(subset))

    @property
    @abstractmethod
    def child_class(self):
        pass


class LeafNode(Node, metaclass=ABCMeta):
    def load(self):
        return self._load()

    def _load(self):
        files = (item for item in os.listdir(self.dir_path))
        files = sorted(files, key=lambda x: int(x.split('.')[0]))
        try:
            data = pd.concat(
                [
                    pd.read_csv(
                        os.path.join(self.dir_path, f),
                        index_col='time',
                        usecols=['time', 'v'],
                        squeeze=True,
                    )
                    for f in files
                ]
            )
            data.index = pd.to_datetime(data.index, utc=True)
        except:
            print(files)
            print(self.dir_path)
        data = data.astype(float)
        return data.rename(self.path_nodes[1:])  # skip root path node

    def get_len(self):
        return 1
