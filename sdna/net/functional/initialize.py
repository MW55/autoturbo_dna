# -*- coding: utf-8 -*-

import torch

CONSTANT = 0.25     # base float used for the weights


def initialize(method="uniform"):
    """
    Returns the desired function for the initialization of the weights for a linear layer.

    :param method: Name of the method to be used.
    :return: Function that is used.
    """
    if method == "normal":
        return weight_init_normal
    elif method == "uniform":
        return weight_init_uniform
    elif method == "constant":
        return weight_init_constant
    elif method == "xavier_normal":
        return weight_init_xavier_normal
    elif method == "xavier_uniform":
        return weight_init_xavier_uniform
    elif method == "kaiming_normal":
        return weight_init_kaiming_normal
    elif method == "kaiming_uniform":
        return weight_init_kaiming_uniform
    else:
        return None


def weight_init_normal(m):
    """
    Fills the input Tensor with values drawn from the normal distribution.

    :param m: A module where the function is to be applied to the linear layer.
    """
    if type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0.0, CONSTANT)


def weight_init_uniform(m):
    """
    Fills the input Tensor with values drawn from the uniform distribution.

    :param m: A module where the function is to be applied to the linear layer.
    """
    if type(m) == torch.nn.Linear:
        torch.nn.init.uniform_(m.weight, -CONSTANT, CONSTANT)


def weight_init_constant(m):
    """
    Fills the input Tensor with a constant value.

    :param m: A module where the function is to be applied to the linear layer.
    """
    if type(m) == torch.nn.Linear:
        torch.nn.init.constant_(m.weight, CONSTANT)


def weight_init_xavier_normal(m):
    """
    Fills the input Tensor with values according to the method described in Understanding the difficulty of training
    deep feedforward neural networks - Glorot, X. & Bengio, Y. (2010), using a normal distribution.

    :param m: A module where the function is to be applied to the linear layer.
    """
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)


def weight_init_xavier_uniform(m):
    """
    Fills the input Tensor with values according to the method described in Understanding the difficulty of training
    deep feedforward neural networks - Glorot, X. & Bengio, Y. (2010), using a uniform distribution.

    :param m: A module where the function is to be applied to the linear layer.
    """
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)


def weight_init_kaiming_normal(m):
    """
    Fills the input Tensor with values according to the method described in Delving deep into rectifiers:
    Surpassing human-level performance on ImageNet classification - He, K. et al. (2015), using a normal distribution.

    :param m: A module where the function is to be applied to the linear layer.
    """
    if type(m) == torch.nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)


def weight_init_kaiming_uniform(m):
    """
    Fills the input Tensor with values according to the method described in Delving deep into rectifiers:
    Surpassing human-level performance on ImageNet classification - He, K. et al. (2015), using a uniform distribution.

    :param m: A module where the function is to be applied to the linear layer.
    """
    if type(m) == torch.nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight)
