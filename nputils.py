import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg as la

def axes3d():
  fig = plt.figure()
  return fig, fig.add_subplot(111, projection='3d')

def mysincf(x):
  """Numerically stable sinc function (sin(x)/x)"""
  z = x if x != 0 else 1e-16
  return np.sin(z) / z

def mysinc(x):
  """Numerically stable sinc function (sin(x)/x)"""
  z = np.select(x == 0, 1e-16, x)
  return np.sin(z) / z

def skew(x):
  """Create skew-symmetric matrix from x"""
  return np.cross(np.eye(3),x.reshape(1,-1))

def expm(w_hat):
  """Matrix exponential of exponential coordinates for rotations"""
  nrm = np.linalg.norm(w) / np.sqrt(2)
  return np.eye(3) + mysinc(nrm)*w_hat + 0.5*(mysinc(nrm/2.0)*w_hat)**2

def expvm(w):
  """Matrix exponential of exponential coordinates for rotations"""
  nrm = np.linalg.norm(w)
  w_hat = skew(w)
  return np.eye(3) + mysinc(nrm)*w_hat + 0.5*(mysinc(nrm/2.0)*w_hat)**2

def logm(R):
  """Matrix logarithm of rotation matrix"""
  tt = np.minimum(np.maximum((np.trace(R)-1)/2.0, -1), 1)
  theta = mysincf(np.arccos(tt))
  if np.abs(theta) < 1e-10:
    print("THETA")
    print(theta)
    print(np.arccos(tt))
    print(tt)
  return (1.0 / (2.0*theta))*np.array([[R[2,1] - R[1,2],
                                                 R[0,2] - R[2,0],
                                                 R[1,0] - R[0,1]]]).T
def sph_rot(x):
  """Takes unit vector and create rotation matrix from it"""
  x = x.reshape(-1)
  u = normalize(np.random.randn(3))
  R = np.array([
          normalize(np.cross(np.cross(u,x),x)),
          normalize(np.cross(u,x)),
          x,
     ])
  return R

def rot_rand_small(sigma):
  r1 = la.qr(np.eye(3) + sigma*np.random.randn(3,3))[0]
  r2 = np.dot(r1, np.diag(np.diag(np.sign(r1)))) 
  return r2

def normalize(x):
  return x / np.linalg.norm(x)

def dim_norm(X):
  """Norms of the vectors along the last dimention of X"""
  return np.expand_dims(np.sqrt(np.sum(X**2, axis=-1)), axis=-1)

def dim_normalize(X):
  """Return X with vectors along last dimention normalized to unit length"""
  return X / dim_norm(X)

def planer_proj(X):
  return X / np.expand_dims(X[...,-1], axis=-1)
