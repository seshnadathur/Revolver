cimport numpy as np
from libc.math cimport exp

def mult_kx(np.ndarray [np.complex128_t, ndim=3] deltaout, np.ndarray [np.complex128_t, ndim=3] delta,
            np.ndarray [np.float64_t, ndim=1] k, double bias):
    cdef int N = delta.shape[0]
    cdef int ix,iy,iz
    for ix in range(N):
      for iy in range(N):
        for iz in range(N):
          deltaout[ix,iy,iz] = delta[ix,iy,iz] * (-1.0j) * k[ix] / bias
    return deltaout

def mult_ky(np.ndarray [np.complex128_t, ndim=3] deltaout, np.ndarray [np.complex128_t, ndim=3] delta,
            np.ndarray [np.float64_t, ndim=1] k, double bias):
    cdef int N = delta.shape[0]
    cdef int ix,iy,iz
    for ix in range(N):
      for iy in range(N):
        for iz in range(N):
          deltaout[ix,iy,iz] = delta[ix,iy,iz] * (-1.0j) * k[iy] / bias
    return deltaout

def mult_kz(np.ndarray [np.complex128_t, ndim=3] deltaout, np.ndarray [np.complex128_t, ndim=3] delta,
            np.ndarray [np.float64_t, ndim=1] k, double bias):
    cdef int N = delta.shape[0]
    cdef int ix,iy,iz
    for ix in range(N):
      for iy in range(N):
        for iz in range(N):
          deltaout[ix,iy,iz] = delta[ix,iy,iz] * (-1.0j) * k[iz] / bias
    return deltaout

def mult_kr2(np.ndarray [np.complex128_t, ndim=3] deltaout, np.ndarray [np.complex128_t, ndim=3] delta,
             np.ndarray [np.float64_t, ndim=1] kr):
    cdef int N = delta.shape[0]
    cdef int ix,iy,iz
    for ix in range(N):
      krx = kr[ix]
      for iy in range(N):
        kry = kr[iy]
        for iz in range(N):
          krz = kr[iz]
          kr2 = krx*krx + kry*kry + krz*krz
          deltaout[ix,iy,iz] = delta[ix,iy,iz] * exp(-0.5*kr2)
    return deltaout

def divide_k2(np.ndarray [np.complex128_t, ndim=3] deltaout, np.ndarray [np.complex128_t, ndim=3] delta,
              np.ndarray [np.float64_t, ndim=1] k):
    cdef int N = delta.shape[0]
    cdef int ix,iy,iz
    cdef double kx,ky,kz,k2
    for ix in range(N):
      kx = k[ix]
      for iy in range(N):
        ky = k[iy]
        for iz in range(N):
          kz = k[iz]
          k2 = kx*kx + ky*ky + kz*kz
          if(ix + iy + iz > 0):
            deltaout[ix,iy,iz] = delta[ix,iy,iz] / k2
    deltaout[0,0,0] = 0.
    return deltaout

def allocate_gal_cic(
    np.ndarray [np.float64_t, ndim=3] delta,
    np.ndarray [np.float64_t, ndim=1] x,
    np.ndarray [np.float64_t, ndim=1] y, 
    np.ndarray [np.float64_t, ndim=1] z,
    np.ndarray [np.float64_t, ndim=1] w,
    int npart,
    double xmin,
    double ymin,
    double zmin,
    double boxsize,
    int nbins,
    int wrap):

  cdef double xpos,ypos,zpoz
  cdef int i,j,k
  cdef int ix,iy,iz
  cdef int ixp,iyp,izp
  cdef double ddx,ddy,ddz
  cdef double mdx,mdy,mdz
  cdef double weight
  cdef double binsize
  cdef double oneoverbinsize

  binsize = boxsize / nbins
  oneoverbinsize = 1.0 / binsize
  weight = 1.0
  
  for i in range(nbins):
    for j in range(nbins):
      for k in range(nbins):
        delta[i,j,k] = 0.

  for i in range(npart):
    if(w is not None):
      weight = w[i]

    xpos = (x[i] - xmin)*oneoverbinsize
    ypos = (y[i] - ymin)*oneoverbinsize
    zpos = (z[i] - zmin)*oneoverbinsize

    ix = int(xpos)
    iy = int(ypos)
    iz = int(zpos)
    
    ddx = xpos-ix
    ddy = ypos-iy
    ddz = zpos-iz
    
    mdx = (1.0 - ddx)*weight
    mdy = (1.0 - ddy)*weight
    mdz = (1.0 - ddz)*weight

    ixp = ix + 1;
    iyp = iy + 1;
    izp = iz + 1;

    if(wrap):
      if(ixp >= nbins): ixp -= nbins
      if(iyp >= nbins): iyp -= nbins
      if(izp >= nbins): izp -= nbins
    else:
      if(ixp >= nbins):
        ixp = 0
        mdx = 0.0
      if(iyp >= nbins):
        iyp = 0
        mdy = 0.0
      if(izp >= nbins):
        izp = 0
        mdz = 0.0

    delta[ix,  iy,  iz]  += mdx * mdy * mdz
    delta[ixp, iy,  iz]  += ddx * mdy * mdz
    delta[ix,  iyp, iz]  += mdx * ddy * mdz
    delta[ix,  iy,  izp] += mdx * mdy * ddz
    delta[ixp, iyp, iz]  += ddx * ddy * mdz
    delta[ixp, iy,  izp] += ddx * mdy * ddz
    delta[ix,  iyp, izp] += mdx * ddy * ddz
    delta[ixp, iyp, izp] += ddx * ddy * ddz

  return delta

def create_delta_survey(np.ndarray [np.complex128_t, ndim=3] delta, np.ndarray [np.float64_t, ndim=3] rhog,
                        np.ndarray [np.float64_t, ndim=3] rhor, double alpha, double ran_min):

  cdef int N = rhog.shape[0]
  cdef int ix, iy, iz
  for ix in range(N):
    for iy in range(N):
      for iz in range(N):
        if rhor[ix, iy, iz] > ran_min:
          delta[ix, iy, iz] = (rhog[ix, iy, iz] / (alpha * rhor[ix, iy, iz])) - 1. + 0.0j
        else:
          delta[ix, iy, iz] = 0. + 0.0j

  return delta

def create_delta_box(np.ndarray [np.complex128_t, ndim=3] delta, np.ndarray [np.float64_t, ndim=3] rhog,
                        int npart):

  cdef int N = rhog.shape[0]
  cdef int ix, iy, iz
  for ix in range(N):
    for iy in range(N):
      for iz in range(N):
        delta[ix, iy, iz] = (rhog[ix, iy, iz] * N**3) / npart - 1.0 + 0.0j

  return delta