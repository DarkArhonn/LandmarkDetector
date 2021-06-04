import typing as tp
import scipy.io as sio
import numpy as np
import tensorflow as tf
import os


from .base_dataset import BaseDataset
from .sampler import ListSampler

class MHPDataset(BaseDataset):
    def __init__(self,
                 dataset_paths:tp.List[str],
                 input_image_shape,
                 batch_size,
                 buffer_size,
                 n_parallel_cals,
                 shuffle=True
                 ):
        super().__init__(dataset_paths, input_image_shape, batch_size, buffer_size, n_parallel_cals, shuffle)
        self.sampler = ListSampler(self.dataset_paths[0],self.dataset_paths[1],shuffle=shuffle)
        self.n_samples = len(self.sampler)
        self.n_batches = self.n_samples // self.batch_size
        self._dataset = tf.data.Dataset.from_generator(self.sampler,output_shapes=(2,),output_types=tf.string)\
                                        .map(lambda paths: self.load_image(paths),
                                             num_parallel_calls=self.n_parallel_cals,
                                             deterministic=False
                                             )



    def load_image(self,paths:tp.List[str]):
        return [
            tf.cast(tf.io.decode_jpeg(tf.io.read_file(paths[0])), dtype=tf.float32),
            tf.keras.applications.efficientnet.preprocess_input(tf.cast(tf.io.decode_jpeg(tf.io.read_file(paths[1])), dtype=tf.float32))
        ]






