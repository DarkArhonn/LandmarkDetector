import typing as tp


class BaseDataset:
    def __init__(self, dataset_paths: tp.List[str],
                 input_image_shape: tp.List[int],
                 batch_size: int,
                 buffer_size: int,
                 n_parallel_cals,
                 shuffle: bool=True):
        self.dataset_paths = dataset_paths
        self.input_image_shape = input_image_shape
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.batch_size = batch_size
        self._dataset = None
        self.n_parallel_cals = n_parallel_cals


    @property
    def dataset(self):
        return self._dataset

