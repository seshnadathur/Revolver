#ifdef _CUTE_AS_PYTHON_MODULE
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "define.h"
#include "common.h"

Result *global_result = NULL;
Catalog *global_galaxy_catalog = NULL;
Catalog *global_galaxy_catalog2 = NULL;
Catalog *global_random_catalog = NULL;
Catalog *global_random_catalog2 = NULL;

Result *make_empty_result_struct(){
  int n_bins_all = 0, nx = 0, ny = 0, nz = 0;
  if(corr_type==0){
    n_bins_all=nb_dz;
    nx = nb_dz;
  } else if(corr_type==1){
    n_bins_all=nb_theta;
    nx = nb_theta;
  } else if(corr_type==2){
    n_bins_all=nb_r;
    nx = nb_r;
  } else if(corr_type==3){
    n_bins_all=nb_rt*nb_rl;
    nx = nb_rt;
    ny = nb_rl;
  } else if(corr_type==4){
    n_bins_all=nb_r*nb_mu;
    nx = nb_r;
    ny = nb_mu;
  } else if(corr_type==5){
    n_bins_all=nb_red*nb_dz*nb_theta;
    nx = nb_red;
    ny = nb_dz;
    nz = nb_theta;
  } else if(corr_type==6) {
    n_bins_all=nb_theta*((nb_red*(nb_red+1))/2);
    nx = nb_red;
    ny = nb_red;
    nz = nb_theta;
  } else if(corr_type==7){
    n_bins_all=nb_r;
    nx = nb_r;
  } else if(corr_type==8){
    n_bins_all=nb_r*nb_mu;
    nx = nb_r;
    ny = nb_mu;
  }
  
  Result *res = malloc(sizeof(Result));
  res->nx   = nx;
  res->ny   = ny;
  res->nz   = nz;
  res->x    = malloc(sizeof(double)*nx);
  res->y    = malloc(sizeof(double)*ny);
  res->z    = malloc(sizeof(double)*nz);
  res->corr = malloc(sizeof(double)*n_bins_all);
  res->D1D1 = malloc(sizeof(double)*n_bins_all);
  res->D1D2 = malloc(sizeof(double)*n_bins_all);
  res->D1R1 = malloc(sizeof(double)*n_bins_all);
  res->D1R2 = malloc(sizeof(double)*n_bins_all);
  res->D2D2 = malloc(sizeof(double)*n_bins_all);
  res->D2R1 = malloc(sizeof(double)*n_bins_all);
  res->D2R2 = malloc(sizeof(double)*n_bins_all);
  res->R1R1 = malloc(sizeof(double)*n_bins_all);
  res->R1R2 = malloc(sizeof(double)*n_bins_all);
  res->R2R2 = malloc(sizeof(double)*n_bins_all);
  return res;
}

int get_corr_type(){
  return corr_type;
}

void set_result_3d(Result *res, int i, int j, int k, int ind, double x, double y, double z, double corr, 
    double D1D1, double D1D2, double D1R1, double D1R2,
    double D2D2, double D2R1, double D2R2,
    double R1R1, double R1R2,
    double R2R2){
  if(res != NULL){
    if(i >= 0) res->x[i] = x;
    if(j >= 0) res->y[j] = y;
    if(k >= 0) res->z[k] = z;
    res->corr[ind]  = corr;
    res->D1D1[ind]  = D1D1;
    res->D1D2[ind]  = D1D2;
    res->D1R1[ind]  = D1R1;
    res->D1R2[ind]  = D1R2;
    res->D2D2[ind]  = D2D2;
    res->D2R1[ind]  = D2R1;
    res->R1R1[ind]  = R1R1;
    res->R1R2[ind]  = R1R2;
    res->R2R2[ind]  = R2R2;
  }
}

void set_result_2d(Result *res, int i, int j, int ind, double x, double y, double corr, 
    double D1D1, double D1D2, double D1R1, double D1R2,
    double D2D2, double D2R1, double D2R2,
    double R1R1, double R1R2,
    double R2R2){

  set_result_3d(res, i, j, -1, ind, x, y, 0.0, corr, 
      D1D1, D1D2, D1R1, D1R2,
            D2D2, D2R1, D2R2,
                  R1R1, R1R2,
                        R2R2);
}

void set_result(Result *res, int i, double x, double corr, 
    double D1D1, double D1D2, double D1R1, double D1R2,
    double D2D2, double D2R1, double D2R2,
    double R1R1, double R1R2,
    double R2R2){

  set_result_3d(res, i, -1, -1, i, x, 0.0, 0.0, corr, 
      D1D1, D1D2, D1R1, D1R2,
            D2D2, D2R1, D2R2,
                  R1R1, R1R2,
                        R2R2);
}

void free_result_struct(Result *res){
  if(res != NULL){
    free(res->x);
    free(res->y);
    free(res->z);
    free(res->corr);
    free(res->D1D1);
    free(res->D1D2);
    free(res->D1R1);
    free(res->D1R2);
    free(res->D2D2);
    free(res->D2R1);
    free(res->D2R2);
    free(res->R1R1);
    free(res->R1R2);
    free(res->R2R2);
    free(res);
  }
}

Catalog *create_catalog_from_numpy(int n, double *phi, int n1, double *cth, int n2, double *red, int n3, double *weight){
#ifdef _WITH_WEIGHTS
  if(! ((n == n1) && (n1 == n2) && (n2 == n3))){
    print_info("Error: create_catalog_from_numpy inconsistent sizes of the arrays [%i %i %i $i]\n", n, n1, n2, n3); 
#else
  if(! ((n == n1) && (n1 == n2))){
    print_info("Error: create_catalog_from_numpy inconsistent sizes of the arrays [%i %i %i]\n", n, n1, n2); 
#endif
    return NULL;
  }
  Catalog *cat = malloc(sizeof(Catalog));
  cat->np = n;
  cat->red=(double *)my_malloc(cat->np*sizeof(double));
  cat->cth=(double *)my_malloc(cat->np*sizeof(double));
  cat->phi=(double *)my_malloc(cat->np*sizeof(double));
#ifdef _WITH_WEIGHTS
  cat->weight=(double *)my_malloc(cat->np*sizeof(double));
#endif
  int i;
  cat->sum_w = 0;
  cat->sum_w2 = 0;
  for(i = 0; i < n; i++){
    cat->phi[i] = phi[i];
    cat->cth[i] = cth[i];
    cat->red[i] = red[i];
#ifdef _WITH_WEIGHTS
    cat->weight[i] = weight[i];
    cat->sum_w += weight[i];
    cat->sum_w2 += weight[i]*weight[i];
#else
    cat->sum_w += 1;
    cat->sum_w2 += 1;
#endif
  }
  return cat;
}
#endif // _CUTE_AS_PYTHON_MODULE
