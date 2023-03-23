import sys
import numpy as np
from common.layers import MatMul

# sample data
c0 = np.array([[1,0,0,0,0,0,0]])
c1 = np.array([[0,0,1,0,0,0,0]])

# initializing weight
w_in = np.random.randn(7, 3)
w_out = np.random.randn(3, 7)

# create layer
in_layer0 = MatMul(w_in)
in_layer1 = MatMul(w_in)
out_layer = MatMul(w_out)

# forward
h0 = in_layer0.forward(c0)
h1 = in_layer1.forward(c1)
h = 0.5 * (h0+h1)
s = out_layer.forward(h)

print(s)