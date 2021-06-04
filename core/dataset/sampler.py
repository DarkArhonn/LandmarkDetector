import typing as tp
import os
import numpy as np
from pathlib import Path

class ListSampler(object):
    def __init__(self,
                 data_list_heat:str,
                 data_list_image:str,
                 shuffle:bool=True,
                 check:bool=True
                 ):
        self.data_list_heat = data_list_heat
        self.data_list_image = data_list_image

        self.shuffle = shuffle
        with open(data_list_image,"r") as image_list_file, open(data_list_heat,"r") as heat_list_file:
            self.image_list = image_list_file.read()
            self.heat_list = heat_list_file.read()
            self.image_list = [name.strip() for name in self.image_list]
            self.heat_list = [name.strip() for name in self.heat_list]

        if len(self.image_list) != len(self.heat_list):
            raise Exception(f"file lists must be same size! heatmap: {data_list_heat} image: {data_list_image}")
        self.n_samples = len(self.image_list)

        self.indecies = np.arange(0, self.n_samples)

        self._reset()

        if check:
            for heat_name, image_name in zip(self.heat_list,self.image_list):
                if not Path(heat_name).exists() and not Path(image_name).exists():
                    raise FileNotFoundError(f"One of the file could not be found! Heatmap: {heat_name}  Image:{image_name}")


    def _reset(self):
        self.indecies = np.arange(self.n_samples)
        self.pos = 0
        if self.shuffle:
            self.indecies = np.random.permutation(self.indecies)


    def __len__(self):
        return self.n_samples

    def __iter__(self):
        self._reset()
        return self

    def __next__(self):
        if self.pos >= self.n_samples:
            self._reset()
        heat = self.heat_list[self.indecies[self.pos]]
        image = self.image_list[self.indecies[self.pos]]
        return (image, heat)

    def __call__(self):
        return self








