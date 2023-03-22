import torch

def get_li_distance(x1, x2):
    return ((x1-x2).abs()).sum()

def get_l2_distance(x1, x2):
    return ((x1-x2)**2).sum()**.5

def get_infinity_distance(x1, x2):
    return ((x1, x2).abs()).max()

def get_cosine_similarity(x1, x2):
    return (x1 * x2).sum() / ((x1**2).sum()**.5 * (x2**2).sum()**.5)