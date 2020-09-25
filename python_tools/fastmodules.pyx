cimport numpy as np
from libc.math cimport exp
from libc.stdio cimport FILE, fopen, fread, fclose
from libc.stdlib cimport malloc, free


cdef struct PARTICLE:
  int nadj
  int nadj_count
  int *adj

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

def mult_norm(np.ndarray [np.complex128_t, ndim=3] rhoout, np.ndarray [np.complex128_t, ndim=3] rhoin,
             np.ndarray [np.float64_t, ndim=3] norm):
    cdef int N = rhoin.shape[0]
    cdef int ix,iy,iz
    for ix in range(N):
      for iy in range(N):
        for iz in range(N):
          rhoout[ix,iy,iz] = rhoin[ix,iy,iz] * norm[ix,iy,iz]
    return rhoout

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

  cdef double xpos,ypos,zpos
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
    else:
      weight = 1

    xpos = (x[i] - xmin)*oneoverbinsize
    ypos = (y[i] - ymin)*oneoverbinsize
    zpos = (z[i] - zmin)*oneoverbinsize

    ix = int(xpos)
    iy = int(ypos)
    iz = int(zpos)

    ddx = xpos-ix
    ddy = ypos-iy
    ddz = zpos-iz

    mdx = (1.0 - ddx)
    mdy = (1.0 - ddy)
    mdz = (1.0 - ddz)

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
        ddx = 0.0
      if(iyp >= nbins):
        iyp = 0
        ddy = 0.0
      if(izp >= nbins):
        izp = 0
        ddz = 0.0

    delta[ix,  iy,  iz]  += mdx * mdy * mdz * weight
    delta[ixp, iy,  iz]  += ddx * mdy * mdz * weight
    delta[ix,  iyp, iz]  += mdx * ddy * mdz * weight
    delta[ix,  iy,  izp] += mdx * mdy * ddz * weight
    delta[ixp, iyp, iz]  += ddx * ddy * mdz * weight
    delta[ixp, iy,  izp] += ddx * mdy * ddz * weight
    delta[ix,  iyp, izp] += mdx * ddy * ddz * weight
    delta[ixp, iyp, izp] += ddx * ddy * ddz * weight

  return delta

def normalize_delta_survey(np.ndarray [np.complex128_t, ndim=3] delta, np.ndarray [np.float64_t, ndim=3] rhog,
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

def normalize_delta_box(np.ndarray [np.complex128_t, ndim=3] delta, np.ndarray [np.float64_t, ndim=3] rhog,
                        int npart):

  cdef int N = rhog.shape[0]
  cdef int ix, iy, iz
  for ix in range(N):
    for iy in range(N):
      for iz in range(N):
        delta[ix, iy, iz] = (rhog[ix, iy, iz] * N**3) / npart - 1.0 + 0.0j

  return delta

def normalize_rho_survey(np.ndarray [np.float64_t, ndim=3] rho_out, np.ndarray [np.float64_t, ndim=3] rhog,
                        np.ndarray [np.float64_t, ndim=3] rhor, double alpha, double ran_min):

  cdef int N = rhog.shape[0]
  cdef int ix, iy, iz
  for ix in range(N):
    for iy in range(N):
      for iz in range(N):
        if rhor[ix, iy, iz] > ran_min:
          rho_out[ix, iy, iz] = (rhog[ix, iy, iz] / (alpha * rhor[ix, iy, iz]))
        else:
          rho_out[ix, iy, iz] = 0.9e30

  return rho_out

def normalize_rho_box(np.ndarray [np.float64_t, ndim=3] rhog, int npart):

  cdef int N = rhog.shape[0]
  cdef int ix, iy, iz
  for ix in range(N):
    for iy in range(N):
      for iz in range(N):
        rhog[ix, iy, iz] = (rhog[ix, iy, iz] * N**3) / npart

  return rhog

def survey_mask(np.ndarray [np.int_t, ndim=1] mask, np.ndarray [np.float64_t, ndim=3] rhor, double ran_min):

  cdef int N = rhor.shape[0]
  cdef int ix, iy, iz
  for ix in range(N):
    for iy in range(N):
      for iz in range(N):
        if rhor[ix, iy, iz] <= ran_min:
          mask[ix*N*N + iy*N + iz] = 1

  return mask

def survey_cuts_logical(np.ndarray [np.int_t, ndim=1] out, np.ndarray [np.float64_t, ndim=1] veto,
                        np.ndarray [np.float64_t, ndim=1] redshift, double zmin, double zmax):

  cdef int N = redshift.shape[0]
  cdef int i
  for i in range(N):
    if (veto[i] == 1.) and (redshift[i] > zmin) and (redshift[i] < zmax):
      out[i] = 1
    else:
      out[i] = 0

  return out

def voxelvoid_cuts(np.ndarray [np.int_t, ndim=1] select, np.ndarray [np.int_t, ndim=1] mask,
                   np.ndarray [np.float64_t, ndim=2] rawvoids, double min_dens_cut):

  cdef int N = rawvoids.shape[0]
  cdef int i, vox
  for i in range(N):
    vox = int(rawvoids[i, 2])
    if (mask[vox] == 0) and (rawvoids[i, 1] == 0) and (rawvoids[i, 3] < min_dens_cut):
    # if (mask[vox] == 0) and (rawvoids[i, 3] < min_dens_cut):
      select[i] = 1
    else:
      select[i] = 0

  return select

def voxelcluster_cuts(np.ndarray [np.int_t, ndim=1] select, np.ndarray [np.int_t, ndim=1] mask,
                   np.ndarray [np.float64_t, ndim=2] rawclusters, double max_dens_cut):

  cdef int N = rawclusters.shape[0]
  cdef int i, vox
  for i in range(N):
    vox = int(rawclusters[i, 2])
    if (mask[vox] == 0) and (rawclusters[i, 1] == 0) and (rawclusters[i, 3] > max_dens_cut):
      select[i] = 1
    else:
      select[i] = 0

  return select

def get_member_densities(np.ndarray [np.float64_t, ndim=1] member_dens, np.ndarray [np.int_t, ndim=1] voxels,
                         np.ndarray [np.float64_t, ndim=1] rho):

  cdef int N = len(voxels)
  cdef int i
  for i in range(N):
    member_dens[i] = rho[voxels[i]]

  return member_dens
