import tensorflow as tf
import typing as tp
from .dataset import BaseDataset
from .registry import register
class BasePipeline(object):
    def __init__(self,
                 n_steps: int,
                 current_step: int,
                 datasets: tp.Dict[str, tp.Any],
                 models: tf.keras.Model,
                 losses:tp.Any,
                 optimizers:tf.keras.optimizers.Optimizer):
        """
        Defines default interface for training pipeline
        :param n_steps:
        :param current_step:
        :param datasets:
        :param models:
        :param losses:
        :param optimizers:
        """

        self.n_steps = n_steps
        self.current_step = current_step
        self.datasets = datasets
        self.models = models
        self.losses = losses
        self.optimizers = optimizers

    def train_step(self,step):
        raise NotImplementedError(f"train_step32 is not implemented yet")

    def training_loop(self):
        raise NotImplementedError(f"training_loop is not implemented yet")

@register("SingleModel")
class SingleModelPipeline(BasePipeline):
    def __init__(self,
                 n_steps: int,
                 current_step: int,
                 datasets: tp.Dict[str, tp.Any],
                 models: tf.keras.Model,
                 losses:tp.Any,
                 optimizers:tf.keras.optimizers.Optimizer):
        super().__init__(n_steps,
                         current_step,
                         datasets,
                         models,
                         losses,
                         optimizers)

    def train_step(self,step):
