///////////////////////////////////////////////////////////////////////
//                                                                   //
//   Copyright 2012 David Alonso                                     //
//                                                                   //
//                                                                   //
// This file is part of CUTE.                                        //
//                                                                   //
// CUTE is free software: you can redistribute it and/or modify it   //
// under the terms of the GNU General Public License as published by //
// the Free Software Foundation, either version 3 of the License, or //
// (at your option) any later version.                               //
//                                                                   //
// CUTE is distributed in the hope that it will be useful, but       //
// WITHOUT ANY WARRANTY; without even the implied warranty of        //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU //
// General Public License for more details.                          //
//                                                                   //
// You should have received a copy of the GNU General Public License //
// along with CUTE.  If not, see <http://www.gnu.org/licenses/>.     //
//                                                                   //
///////////////////////////////////////////////////////////////////////

#ifndef _CUTE_COMMON_
#define _CUTE_COMMON_

extern int NodeThis;
extern int NNodes;
void mpi_init(int* p_argc,char*** p_argv);
void share_iters(int n_iters,int *iter0,int *niter_this);

//General-purpose functions
void print_info(char *fmt,...);

void *my_malloc(size_t size);

void *my_calloc(size_t nmemb,size_t size);

int linecount(FILE *f);

void timer(int i);

double wrap_phi(double phi);

void error_open_file(char *fname);

void error_read_line(char *fname,int nlin);

void free_Catalog(Catalog *cat);

void free_Catalog_f(Catalog_f cat);


//2D boxes
void free_Cells2D(int npix,Cell2D *cells);

void free_Boxes2D(int npix,Box2D *boxes);

void free_RadialPixels(int npix,RadialPixel *pixrad);

void free_RadialCells(int npix,RadialCell *radcell);

void init_2D_params(Catalog *cat_dat,Catalog *cat_ran,int ctype);

Cell2D *mk_Cells2D_from_Catalog(Catalog *cat,int **cell_indices,
				int *n_cell_full);

Cell2D *mk_Cells2D_many_from_Catalog(Catalog *cat,int **cell_indices,
				     Cell2D **cells_total_out,int *n_cell_full);

RadialCell *mk_RadialCells_from_Catalog(Catalog *cat);

Box2D *mk_Boxes2D_from_Catalog(Catalog *cat,int **box_indices,
			       int *n_box_full);

RadialPixel *mk_RadialPixels_from_Catalog(Catalog *cat,int **pixrad_indices,
					  int *n_pixrad_full,int ctype);

void mk_Cells2D_from_Catalog_f(Catalog_f cat_dat,Catalog_f cat_ran,
			       int *npix,int **pix_full,
			       int **pix_dat,int **pix_ran,float **pix_pos);

void init_2D_params_f(float *cth_min,float *cth_max,
		      Catalog_f cat_dat,Catalog_f cat_ran);

void mk_Boxes2D_from_Catalog_f(Catalog_f cat,float **box_pos,
			       int **box_np,int **box_ind);


//3D boxes
void free_Boxes3D(int nbox,Box3D *boxes);

void init_3D_params(Catalog *cat_dat,Catalog *cat_ran,int ctype);

void init_3D_params_cross(Catalog *cat_dat1,Catalog *cat_dat2,Catalog *cat_ran1,Catalog *cat_ran2,int ctype);

Box3D *mk_Boxes3D_from_Catalog(Catalog *cat,int **box_indices,int *n_box_full);

void init_3D_params_f(float pox_min[],Catalog_f cat_dat,Catalog_f cat_ran,int ctype);

void mk_Boxes3D_from_Catalog_f(Catalog_f cat,float **box_pos,
			       int **box_np,int **box_ind);


//Distance-redshift relation
void end_r_z(void);

double z2r(double zz);

void set_r_z(void);


//Mask and redshift distribution for randoms
void read_red_dist(void);

void read_mask(void);

void end_mask(void);

Catalog *mk_random_cat(int np);

Catalog_f mk_random_cat_f(int np);


//I/O functions
void write_CF(char *fname,histo_t *DD,histo_t *DR,histo_t *RR,
	      np_t sum_wd,np_t sum_wd2,np_t sum_wr,np_t sum_wr2);

void write_CCF(char *fname,histo_t *D1D2,histo_t *D1R,histo_t *D2R,histo_t *RR,
	      np_t sum_wd1,np_t sum_wd2,np_t sum_wr1,np_t sum_wr2,int reuse_ran);

void write_CF_cuda(char *fname,unsigned long long *DD,
		   unsigned long long *DR,unsigned long long *RR,
		   int nD,int nR);

void read_run_params(char *fname);

Catalog *read_catalog(char *fname,np_t *sum_w,np_t *sum_w2);

Catalog_f read_catalog_f(char *fname,int *np);


//Correlators
void auto_angular_cross_bf(int npix_full,int *indices,
			   RadialPixel *pixrad,histo_t *hh);
void cross_angular_cross_bf(int npix_full,int *indices,
			    RadialPixel *pixrad1,RadialPixel *pixrad2,
			    histo_t *hh);
void corr_angular_cross_pm(Cell2D *cellsD,Cell2D *cellsD_total,
			   Cell2D *cellsR,Cell2D *cellsR_total,
			   histo_t *DD,histo_t *DR,histo_t *RR);

void auto_full_bf(int npix_full,int *indices,RadialPixel *pixrad,
		 histo_t *hh);
void cross_full_bf(int npix_full,int *indices,
		   RadialPixel *pixrad1,RadialPixel *pixrad2,
		   histo_t *hh);
void corr_full_pm(RadialCell *cellsD,RadialCell *cellsR,
		 histo_t *DD,histo_t *DR,
		 histo_t *RR);

void auto_rad_bf(int npix_full,int *indices,RadialPixel *pixrad,
		 histo_t *hh);
void cross_rad_bf(int npix_full,int *indices,
		  RadialPixel *pixrad1,RadialPixel *pixrad2,
		  histo_t *hh);

void auto_ang_bf(int npix_full,int *indices,Box2D *boxes,
		 histo_t *hh);
void cross_ang_bf(int npix_full,int *indices,
		  Box2D *boxes1,Box2D *boxes2,
		  histo_t *hh);
void corr_ang_pm(Cell2D *cellsD,Cell2D *cellsR,
		 histo_t *DD,histo_t *DR,
		 histo_t *RR);

void auto_mono_bf(int nbox_full,int *indices,Box3D *boxes,
		  histo_t *hh);
void cross_mono_bf(int nbox_full,int *indices,
		   Box3D *boxes1,Box3D *boxes2,
		   histo_t *hh);

void auto_3d_ps_bf(int nbox_full,int *indices,Box3D *boxes,
		   histo_t *hh);
void cross_3d_ps_bf(int nbox_full,int *indices,
		    Box3D *boxes1,Box3D *boxes2,
		    histo_t *hh);

void auto_3d_rm_bf(int nbox_full,int *indices,Box3D *boxes,
		   histo_t *hh);
void auto_3d_rm_special_bf(int nbox_full,int *indices,Box3D *boxes,
		   histo_t *hh);
void cross_3d_rm_bf(int nbox_full,int *indices,
		    Box3D *boxes1,Box3D *boxes2,
		    histo_t *hh);
void cross_3d_rm_special_bf(int nbox_full,int *indices,
		    Box3D *boxes1,Box3D *boxes2,
		    histo_t *hh);

#ifdef _DEBUG
//Debug files output
void write_Cells2D(int num_cells,Cell2D *cellmap,char *fn);

void write_RadialCells(int num_cells,RadialCell *cellmap,char *fn);

void write_Boxes2D(int num_boxes,Box2D *boxes,char *fn);

void write_PixRads(int num_pix,RadialPixel *pixrad,char *fn);

void write_Boxes3D(int num_boxes,Box3D *boxes,char *fn);

void write_Catalog(Catalog *cat,char *fn);

void write_Catalog_f(Catalog_f cat,char *fn);
#endif //_DEBUG

#ifdef _CUTE_AS_PYTHON_MODULE

void print_parameters();

Catalog *read_Catalog(char *fname);

int get_corr_type();

Result *make_empty_result_struct();
void free_result_struct(Result *res);
void set_result(Result *res, int i, double x, double corr, 
    double D1D1, double D1D2, double D1R1, double D1R2,
    double D2D2, double D2R1, double D2R2,
    double R1R1, double R1R2,
    double R2R2);
void set_result_2d(Result *res, int i, int j, int ind, double x, double y, double corr, 
    double D1D1, double D1D2, double D1R1, double D1R2,
    double D2D2, double D2R1, double D2R2,
    double R1R1, double R1R2,
    double R2R2);
void set_result_3d(Result *res, 
    int i, int j, int k, int ind, 
    double x, double y, double z, double corr, 
    double D1D1, double D1D2, double D1R1, double D1R2,
    double D2D2, double D2R1, double D2R2,
    double R1R1, double R1R2,
    double R2R2);

Catalog *create_catalog_from_numpy(int n, double *phi, int n1, double *cth, int n2, double *red, int n3, double *weight);
    
void initialize_binner();
int verify_parameters();
void set_data_filename(char *s);
void set_data_filename2(char *s);
void set_random_filename(char *s);
void set_random_filename2(char *s);
void set_reuse_randoms(int i);
void set_num_lines(char *s);
void set_input_format(int i);
void set_output_filename(char *s);
void set_mask_filename(char *s);
void set_z_dist_filename(char *s);
void set_corr_estimator(char *s);
void set_corr_type(char *s);
void set_np_rand_fact(int i);
void set_omega_M(double x);
void set_omega_L(double x);
void set_w(double x);
void set_radial_aperture(double x);
void set_dim1_max(double x);
void set_dim2_max(double x);
void set_dim3_max(double x);
void set_dim3_min(double x);
void set_dim1_nbin(int i);
void set_dim2_nbin(int i);
void set_dim3_nbin(int i);
void set_log_bin(int i);
void set_n_logint(int i);
void set_use_pm(int i);
void set_n_pix_sph(int i);

#endif

#endif //_CUTE_COMMON_
