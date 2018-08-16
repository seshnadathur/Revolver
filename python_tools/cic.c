#include<stdio.h>
#include<stdlib.h>

// To avoid it wrapping around the box use #define _DONTWRAPAROUND

void add_to_grid_CIC_3D(double *grid, double x, double y, double z, int nx, int ny, int nz, double w){
  int ix = (int)(x * nx);
  int iy = (int)(y * ny);
  int iz = (int)(z * nz);

  double DX = x * nx - (double)ix;
  double DY = y * ny - (double)iy;
  double DZ = z * nz - (double)iz;

  double TX = 1.0 - DX;
  double TY = 1.0 - DY;
  double TZ = 1.0 - DZ;

  if(ix >= nx) ix -= nx;
  if(iy >= ny) iy -= ny;
  if(iz >= nz) iz -= nz;

int ixneigh = ix+1;
int iyneigh = iy+1;
int izneigh = iz+1;

  if(ixneigh >= nx) {
    ixneigh -= nx;
#ifdef _DONTWRAPAROUND
    DX = 0.0;
#endif
  }
  if(iyneigh >= ny) {
    iyneigh -= ny;
#ifdef _DONTWRAPAROUND
    DY = 0.0;
#endif
  }
  if(izneigh >= nz) {
    izneigh -= nz;
#ifdef _DONTWRAPAROUND
    DZ = 0.0;
#endif
  }

  // Multiply by weight
  TX *= w;
  DX *= w;

  grid[iz      + nz*(iy      +ny*ix     )] += TX*TY*TZ;
  grid[izneigh + nz*(iy      +ny*ix     )] += TX*TY*DZ;
  grid[iz      + nz*(iyneigh +ny*ix     )] += TX*DY*TZ;
  grid[izneigh + nz*(iyneigh +ny*ix     )] += TX*DY*DZ;
  grid[iz      + nz*(iy      +ny*ixneigh)] += DX*TY*TZ;
  grid[izneigh + nz*(iy      +ny*ixneigh)] += DX*TY*DZ;
  grid[iz      + nz*(iyneigh +ny*ixneigh)] += DX*DY*TZ;
  grid[izneigh + nz*(iyneigh +ny*ixneigh)] += DX*DY*DZ;
}

void add_to_grid_CIC_2D(double *grid, double x, double y, int nx, int ny, double w){
  int ix = (int)(x * nx);
  int iy = (int)(y * ny);

  double DX = x * nx - (double)ix;
  double DY = y * ny - (double)iy;

  double TX = 1.0 - DX;
  double TY = 1.0 - DY;

  if(ix >= nx) ix -= nx;
  if(iy >= ny) iy -= ny;

  int ixneigh = ix+1;
  int iyneigh = iy+1;

  if(ixneigh >= nx) {
    ixneigh -= nx;
#ifdef _DONTWRAPAROUND
    DX = 0.0;
#endif
  }
  if(iyneigh >= ny) {
    iyneigh -= ny;
#ifdef _DONTWRAPAROUND
    DY = 0.0;
#endif
  }
  
  // Multiply by weight
  TX *= w;
  DX *= w;
  
  grid[iy      +ny*ix     ] += TX*TY;
  grid[iyneigh +ny*ix     ] += TX*DY;
  grid[iy      +ny*ixneigh] += DX*TY;
  grid[iyneigh +ny*ixneigh] += DX*DY;
}

void add_to_grid_CIC_1D(double *grid, double x, int nx, double w){
  int ix = (int)(x * nx);

  double DX = x * nx - (double)ix;

  double TX = 1.0 - DX;

  if(ix >= nx) ix -= nx;

  int ixneigh = ix+1;

  if(ixneigh >= nx) {
    ixneigh -= nx;
#ifdef _DONTWRAPAROUND
    DX = 0.0;
#endif
  }
  
  // Multiply by weight
  TX *= w;
  DX *= w;
  
  grid[ix     ] += TX;
  grid[ixneigh] += DX;
}

void perform_cic_3D_w(int n1, int n2, int n3, double *grid, int ix, double *x, int iy, double *y, int iz, double *z, int iw, double *w){
  int n = ix;
  if(ix != iy || iy != iz || iz != iw){
    printf("Error: unmatching dimensions for positions %i %i %i %i\n",ix,iy,iz,iw);
    return;
  }

  for(int i = 0; i < n; i++){
    add_to_grid_CIC_3D(grid, x[i], y[i], z[i], n1, n2, n3, w[i]);
  }
}

void perform_cic_2D_w(int n1, int n2, double *grid, int ix, double *x, int iy, double *y, int iw, double *w){
  int n = ix;
  if(ix != iy || iy != iw){
    printf("Error: unmatching dimensions for positions %i %i %i\n",ix,iy,iw);
    return;
  }

  for(int i = 0; i < n; i++){
    add_to_grid_CIC_2D(grid, x[i], y[i], n1, n2, w[i]);
  }
}

void perform_cic_1D_w(int n1, double *grid, int ix, double *x, int iw, double *w){
  int n = ix;
  if(ix != iw){
    printf("Error: unmatching dimensions for positions %i %i\n",ix,iw);
    return;
  }

  for(int i = 0; i < n; i++){
    add_to_grid_CIC_1D(grid, x[i], n1, w[i]);
  }
}

void perform_cic_3D(int n1, int n2, int n3, double *grid, int ix, double *x, int iy, double *y, int iz, double *z){
  int n = ix;
  if(ix != iy || iy != iz){
    printf("Error: unmatching dimensions for positions %i %i %i\n",ix,iy,iz);
    return;
  }

  for(int i = 0; i < n; i++){
    add_to_grid_CIC_3D(grid, x[i], y[i], z[i], n1, n2, n3, 1.0);
  }
}

void perform_cic_2D(int n1, int n2, double *grid, int ix, double *x, int iy, double *y){
  int n = ix;
  if(ix != iy){
    printf("Error: unmatching dimensions for positions %i %i\n",ix,iy);
    return;
  }

  for(int i = 0; i < n; i++){
    add_to_grid_CIC_2D(grid, x[i], y[i], n1, n2, 1.0);
  }
}

void perform_cic_1D(int n1, double *grid, int ix, double *x){
  int n = ix;

  for(int i = 0; i < n; i++){
    add_to_grid_CIC_1D(grid, x[i], n1, 1.0);
  }
}


