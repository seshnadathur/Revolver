%module cic
%{
  #define SWIG_FILE_WITH_INIT
  extern void perform_cic_3D_w(int n1, int n2, int n3, double *grid, int ix, double *x, int iy, double *y, int iz, double *z, int iw, double *w);
  extern void perform_cic_2D_w(int n1, int n2, double *grid, int ix, double *x, int iy, double *y, int iw, double *w);
  extern void perform_cic_1D_w(int n1, double *grid, int ix, double *x, int iw, double *w);
  
  extern void perform_cic_3D(int n1, int n2, int n3, double *grid, int ix, double *x, int iy, double *y, int iz, double *z);
  extern void perform_cic_2D(int n1, int n2, double *grid, int ix, double *x, int iy, double *y);
  extern void perform_cic_1D(int n1, double *grid, int ix, double *x);
%}
  
%include "numpy.i"
%init %{
  import_array();
%}

/* Rules for allowing calls with numpy arrays */
%apply (int DIM1, double* IN_ARRAY1) {(int ix, double *x)};
%apply (int DIM1, double* IN_ARRAY1) {(int iy, double *y)};
%apply (int DIM1, double* IN_ARRAY1) {(int iz, double *z)};
%apply (int DIM1, double* IN_ARRAY1) {(int iw, double *w)};
%apply (int DIM1, double* INPLACE_ARRAY1) {(int n1, double *grid)};
%apply (int DIM1, int DIM2, double* INPLACE_ARRAY2) {(int n1, int n2, double *grid)};
%apply (int DIM1, int DIM2, int DIM3, double* INPLACE_ARRAY3) {(int n1, int n2, int n3, double *grid)};

extern void perform_cic_3D_w(int n1, int n2, int n3, double *grid, int ix, double *x, int iy, double *y, int iz, double *z, int iw, double *w);
extern void perform_cic_2D_w(int n1, int n2, double *grid, int ix, double *x, int iy, double *y, int iw, double *w);
extern void perform_cic_1D_w(int n1, double *grid, int ix, double *x, int iw, double *w);
  
extern void perform_cic_3D(int n1, int n2, int n3, double *grid, int ix, double *x, int iy, double *y, int iz, double *z);
extern void perform_cic_2D(int n1, int n2, double *grid, int ix, double *x, int iy, double *y);
extern void perform_cic_1D(int n1, double *grid, int ix, double *x);
