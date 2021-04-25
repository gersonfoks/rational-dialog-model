'''
Small demo on how the kurwaswamy distribtution works.
'''

import torch

# Set parameters for the Rectified Kumaraswamy distribution
l = -0.1
r = 1.1
a = 0.5
b = 0.5

def inverse_kuma(u, a, b):
    return (1 - (1 - u)**(1/b))**(1/a)

def stretch(k, l, r):
    return l + (r - l) * k

def rectify(t):
    maxed = t < 0
    t_maxed = (t * ~maxed) + torch.zeros(t.shape) * maxed
    mined = t_maxed > 1
    h = (t_maxed * ~mined) + torch.ones(t_maxed.shape) * mined
    return h 

# Now we randomly sample from it.
probs = torch.rand(1000)
k = inverse_kuma(probs, a, b)

# Stretch shift and scale the result
t = stretch(k, l, r)

# Restrict to interval [0-1]
h = rectify(t)

print("Uniform sample: ", torch.histc(probs, bins=10, min=0.0, max=1.0), min(probs), max(probs))
print("Inverse Kuma:   ", torch.histc(k, bins=10, min=0.0, max=1.0), min(k), max(k))
print("Rectified Kuma: ", torch.histc(h, bins=10, min=0.0, max=1.0), min(h), max(h))
