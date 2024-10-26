""" Special functions for GELU activation.

Attempted to estimate E[gelu(x)^2] with x ~ N(0, 1) for VP features in our activation.

Needed for stable hypernetwork training.
"""
import numpy as np
import scipy.integrate as integrate
from scipy.stats import norm

# use module to calculate
GELU_E2 = 0.42522148257029874
rv = norm()


def gelu_act(x):
  """ Activation function GELU """
  return x * rv.cdf(x)  # 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715*(x*x*x))))


def normal_pdf(x, mean, std):
  """ Gets the pdf value for a normal distribution with params mean, std """
  xn = ((x - mean)/std)
  return (1.0 / (std*np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * xn * xn)


def normal_gelu_cov(x):
  """ The function change of variable for E[gelu^2(X)] with X ~ N(0, 1) """
  gv = gelu_act(x)
  return (gv*gv) * normal_pdf(x, 0.0, 100.0)


if __name__ == '__main__':
  print('Performing GELU E[gelu(x)^2] integration')
  print('Result:', integrate.quad(normal_gelu_cov, -800, 800, epsabs=1e-12, epsrel=1e-12)[0]/10000.0)
