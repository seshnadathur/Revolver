%module CUTEPython
%{
  #define SWIG_FILE_WITH_INIT
  #include "src/define.h"
  #include "src/common.h"
  extern int runCUTE(Catalog *galaxy_catalog, Catalog *galaxy_catalog2, Catalog *random_catalog, Catalog *random_catalog2, Result *result);
  extern Catalog *read_Catalog(char *fname);
  extern void free_Catalog(Catalog *cat);
  extern void read_run_params(char *paramfile);
  extern Result *make_empty_result_struct();
  extern void free_result_struct(Result *res);
  extern int get_corr_type();
  extern void finalize_mpi();

  extern Catalog *create_catalog_from_numpy(int n, double *phi, int n1, double *cth, int n2, double *red, int n3, double *weight);

  extern void initialize_binner();
  extern int verify_parameters();
  extern void print_parameters();

  extern void set_data_filename(char *s);
  extern void set_data_filename2(char *s);
  extern void set_random_filename(char *s);
  extern void set_random_filename2(char *s);
  extern void set_reuse_randoms(int i);
  extern void set_num_lines(char *s);
  extern void set_input_format(int i);
  extern void set_output_filename(char *s);
  extern void set_mask_filename(char *s);
  extern void set_z_dist_filename(char *s);
  extern void set_corr_estimator(char *s);
  extern void set_corr_type(char *s);
  extern void set_np_rand_fact(int i);
  extern void set_omega_M(double x);
  extern void set_omega_L(double x);
  extern void set_w(double x);
  extern void set_radial_aperture(double x);
  extern void set_dim1_max(double x);
  extern void set_dim2_max(double x);
  extern void set_dim3_max(double x);
  extern void set_dim3_min(double x);
  extern void set_dim1_nbin(int i);
  extern void set_dim2_nbin(int i);
  extern void set_dim3_nbin(int i);
  extern void set_log_bin(int i);
  extern void set_n_logint(int i);
  extern void set_use_pm(int i);
  extern void set_n_pix_sph(int i);

  struct Catalog{
    int np;
    double *red,*cth,*phi;
  #ifdef _WITH_WEIGHTS
    double *weight;
  #endif
    np_t sum_w, sum_w2;
  };

  struct Result {
    int nx, ny, nz;
    double *x, *y, *z, *corr, 
         *D1D1, *D1D2, *D1R1, *D1R2,
         *D2D2, *D2R1, *D2R2, 
         *R1R1, *R1R2, 
         *R2R2;
  };
%}

%include "numpy.i"
%init %{
import_array();
%}
%apply (int DIM1, double* INPLACE_ARRAY1) {(int n0, double *a0)};
%apply (int DIM1, double* IN_ARRAY1) {(int n, double *phi), (int n1, double *cth), (int n2, double *red), (int n3, double *weight)};

#include "src/define.h"
#include "src/common.h"
extern int runCUTE(Catalog *galaxy_catalog, Catalog *galaxy_catalog2, Catalog *random_catalog, Catalog *random_catalog2, Result *result);
extern Catalog *read_Catalog(char *fname);
extern void free_Catalog(Catalog *cat);
extern void read_run_params(char *paramfile);
extern Result *make_empty_result_struct();
extern void free_result_struct(Result *res);
extern int get_corr_type();
extern void finalize_mpi();

extern Catalog *create_catalog_from_numpy(int n, double *phi, int n1, double *cth, int n2, double *red, int n3, double *weight);

extern void initialize_binner();
extern int verify_parameters();
extern void print_parameters();

extern void set_data_filename(char *s);
extern void set_data_filename2(char *s);
extern void set_random_filename(char *s);
extern void set_random_filename2(char *s);
extern void set_reuse_randoms(int i);
extern void set_num_lines(char *s);
extern void set_input_format(int i);
extern void set_output_filename(char *s);
extern void set_mask_filename(char *s);
extern void set_z_dist_filename(char *s);
extern void set_corr_estimator(char *s);
extern void set_corr_type(char *s);
extern void set_np_rand_fact(int i);
extern void set_omega_M(double x);
extern void set_omega_L(double x);
extern void set_w(double x);
extern void set_radial_aperture(double x);
extern void set_dim1_max(double x);
extern void set_dim2_max(double x);
extern void set_dim3_max(double x);
extern void set_dim3_min(double x);
extern void set_dim1_nbin(int i);
extern void set_dim2_nbin(int i);
extern void set_dim3_nbin(int i);
extern void set_log_bin(int i);
extern void set_n_logint(int i);
extern void set_use_pm(int i);
extern void set_n_pix_sph(int i);

struct Catalog{
  int np;
  double *red,*cth,*phi;
#ifdef _WITH_WEIGHTS
  double *weight;
#endif
  np_t sum_w, sum_w2;
};

%extend Catalog{
  int get_np(){
    return $self->np;
  }
  double get_red(int i) {
    return $self->red[i];
  }
  double get_cth(int i) {
    return $self->cth[i];
  }
  double get_phi(int i) {
    return $self->phi[i];
  }
#ifdef _WITH_WEIGHTS
  double get_weight(int i) {
    return $self->weight[i];
  }
#endif
  np_t get_sum_w(){
    return $self->sum_w;
  }
  np_t get_sum_w2(){
    return $self->sum_w2;
  }
}

%typemap(newfree) Catalog * {
  free_Catalog($1);
}

struct Result {
  int nx, ny, nz;
  double *x, *y, *z, *corr, 
         *D1D1, *D1D2, *D1R1, *D1R2,
         *D2D2, *D2R1, *D2R2, 
         *R1R1, *R1R2, 
         *R2R2;
};

%extend Result{
  int get_nx(){
    return $self->nx;
  }
  int get_ny(){
    return $self->ny;
  }
  int get_nz(){
    return $self->nz;
  }
  double get_x(int i) {
    if(self->nx > 0)
      return $self->x[i];
    return 0.0;
  }
  double get_y(int i) {
    if(self->ny > 0)
      return $self->y[i];
    return 0.0;
  }
  double get_z(int i) {
    if(self->nz > 0)
      return $self->z[i];
    return 0.0;
  }
  double get_corr(int i) {
    return $self->corr[i];
  }
  double get_D1D1(int i) {
    return $self->D1D1[i];
  }
  double get_D1D2(int i) {
    return $self->D1D2[i];
  }
  double get_D1R1(int i) {
    return $self->D1R1[i];
  }
  double get_D1R2(int i) {
    return $self->D1R2[i];
  }
  double get_D2D2(int i) {
    return $self->D2D2[i];
  }
  double get_D2R1(int i) {
    return $self->D2R1[i];
  }
  double get_D2R2(int i) {
    return $self->D2R2[i];
  }
  double get_R1R1(int i) {
    return $self->R1R1[i];
  }
  double get_R1R2(int i) {
    return $self->R1R2[i];
  }
  double get_R2R2(int i) {
    return $self->R2R2[i];
  }
  ~Result(){
    free($self->x);
    free($self->y);
    free($self->z);
    free($self->corr);
    free($self->D1D1);
    free($self->D1D2);
    free($self->D1R1);
    free($self->D1R2);
    free($self->D2D2);
    free($self->D2R1);
    free($self->D2R2);
    free($self->R1R1);
    free($self->R1R2);
    free($self->R2R2);
  }
}

%typemap(newobject) make_empty_result_struct;
%typemap(newfree) Result * {
  free_result_struct($1);
}
