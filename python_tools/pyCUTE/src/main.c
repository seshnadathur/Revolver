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

/*********************************************************************/
//                               Main                                //
/*********************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#ifdef _HAVE_OMP
#include <omp.h>
#endif //_HAVE_OMP
#include "define.h"
#include "common.h"

void read_dr_catalogs(Catalog **cat_d,Catalog **cat_r,
    np_t *sum_wd,np_t *sum_wd2,
    np_t *sum_wr,np_t *sum_wr2)
{
  //////
  // Reads or creates random and data catalogs
  Catalog *cat_dat, *cat_ran;

#ifdef _CUTE_AS_PYTHON_MODULE
  if(global_galaxy_catalog == NULL){
    cat_dat=read_catalog(fnameData,sum_wd,sum_wd2);
  } else {
    cat_dat=global_galaxy_catalog;
    *sum_wd = cat_dat->sum_w;
    *sum_wd2 = cat_dat->sum_w2;
  }
#else
  cat_dat=read_catalog(fnameData,sum_wd,sum_wd2);
#endif
  if(gen_ran) {
    read_mask();
    if(corr_type!=1)
      read_red_dist();
    timer(0);
    cat_ran=mk_random_cat(fact_n_rand*(cat_dat->np));
    timer(1);
    end_mask();
    *sum_wr=(np_t)(fact_n_rand*(cat_dat->np));
    *sum_wr2=(np_t)(fact_n_rand*(cat_dat->np));
  }
  else{
#ifdef _CUTE_AS_PYTHON_MODULE
    if(global_random_catalog == NULL){
      cat_ran=read_catalog(fnameRandom,sum_wr,sum_wr2);
    } else {
      cat_ran=global_random_catalog;
      *sum_wr = cat_ran->sum_w;
      *sum_wr2 = cat_ran->sum_w2;
    }
#else
    cat_ran=read_catalog(fnameRandom,sum_wr,sum_wr2);
#endif
  }

#ifdef _DEBUG
  write_Catalog(cat_d,"debug_DatCat.dat");
  write_Catalog(cat_r,"debug_RanCat.dat");
#endif //_DEBUG

  *cat_d=cat_dat;
  *cat_r=cat_ran;
}

void read_ddrr_catalogs(Catalog **cat_d1,Catalog **cat_d2,Catalog **cat_r1,Catalog **cat_r2,
    np_t *sum_wd1,np_t *sum_wd2,np_t *sum_wr1,np_t *sum_wr2,np_t *junk)
{
  //////
  // Reads or creates random and data catalogs
  Catalog *cat_dat1, *cat_dat2;
  Catalog *cat_ran1, *cat_ran2;

#ifdef _CUTE_AS_PYTHON_MODULE
  if(global_galaxy_catalog == NULL){
    cat_dat1=read_catalog(fnameData,sum_wd1,junk);
  } else {
    cat_dat1=global_galaxy_catalog;
    *sum_wd1 = cat_dat1->sum_w;
  }
  if(global_galaxy_catalog2 == NULL){
    cat_dat2=read_catalog(fnameData2,sum_wd2,junk);
  } else {
    cat_dat2=global_galaxy_catalog2;
    *sum_wd2 = cat_dat2->sum_w;
  }
  if(global_random_catalog == NULL){
    cat_ran1=read_catalog(fnameRandom,sum_wr1,junk);
  } else {
    cat_ran1=global_random_catalog;
    *sum_wr1 = cat_ran1->sum_w;
  }
  if(global_random_catalog2 == NULL){
    cat_ran2=read_catalog(fnameRandom2,sum_wr2,junk);
  } else {
    cat_ran2=global_random_catalog2;
    *sum_wr2 = cat_ran2->sum_w;
  }
#else
  cat_dat1=read_catalog(fnameData,sum_wd1,junk);
  cat_dat2=read_catalog(fnameData2,sum_wd2,junk);
  cat_ran1=read_catalog(fnameRandom,sum_wr1,junk);
  cat_ran2=read_catalog(fnameRandom2,sum_wr2,junk);
#endif

#ifdef _DEBUG
  write_Catalog(cat_dat1,"debug_DatCat1.dat");
  write_Catalog(cat_dat2,"debug_DatCat2.dat");
  write_Catalog(cat_ran1,"debug_RanCat1.dat");
  write_Catalog(cat_ran2,"debug_RanCat2.dat");
#endif //_DEBUG

  *cat_d1=cat_dat1;
  *cat_d2=cat_dat2;
  *cat_r1=cat_ran1;
  *cat_r2=cat_ran2;
}

void run_angular_cross_corr_bf(void)
{
  //////
  // Runs xi(theta,dz,z) in brute-force mode
  np_t sum_wd,sum_wd2,sum_wr,sum_wr2;
  Catalog *cat_dat,*cat_ran;

  RadialPixel *pixrad_dat,*pixrad_ran;
  int *indices_dat,*indices_ran;
  int nfull_dat,nfull_ran;

  histo_t *DD=(histo_t *)my_calloc(nb_red*(nb_red+1)*nb_theta/2,sizeof(histo_t));
  histo_t *DR=(histo_t *)my_calloc(nb_red*(nb_red+1)*nb_theta/2,sizeof(histo_t));
  histo_t *RR=(histo_t *)my_calloc(nb_red*(nb_red+1)*nb_theta/2,sizeof(histo_t));

  timer(4);

#ifdef _VERBOSE
  print_info("*** Angular cross-correlations: \n");
  print_info(" - Redshift range: %.3lf < z_mean < %.3lf\n",
      red_0,red_0+1./i_red_interval);
  print_info(" - # redshift bins: %d\n",nb_red);
  print_info(" - Redshift bin width: %.3lf\n",1/(i_red_interval*nb_red));

  print_info(" - Angular range: %.3lf < theta < %.3lf \n",
      0.,1./(i_theta_max*DTORAD));
  print_info(" - # angular bins : %d\n",nb_theta);
  if(logbin) {
    print_info(" - Logarithmic binning with %d bins per decade\n",
        n_logint);
  }
  else {
    print_info(" - Resolution: D(theta) = %.3lf \n",
        1./(i_theta_max*nb_theta*DTORAD));
  }

  print_info(" - Using a brute-force approach \n");
  print_info("\n");
#endif

  read_dr_catalogs(&cat_dat,&cat_ran,
      &sum_wd,&sum_wd2,&sum_wr,&sum_wr2);

  print_info("*** Boxing catalogs \n");
  init_2D_params(cat_dat,cat_ran,5);
  pixrad_dat=mk_RadialPixels_from_Catalog(cat_dat,&indices_dat,&nfull_dat,5);
#ifdef _CUTE_AS_PYTHON_MODULE
  if(global_galaxy_catalog == NULL)
#endif
    free_Catalog(cat_dat);
  pixrad_ran=mk_RadialPixels_from_Catalog(cat_ran,&indices_ran,&nfull_ran,5);
#ifdef _CUTE_AS_PYTHON_MODULE
  if(global_random_catalog == NULL)
#endif
    free_Catalog(cat_ran);
  print_info("\n");

#ifdef _DEBUG
  write_PixRads(n_boxes2D,pixrad_dat,"debug_PixRadDat.dat");
  write_PixRads(n_boxes2D,pixrad_ran,"debug_PixRadRan.dat");
#endif //_DEBUG

  print_info("*** Correlating \n");
  print_info(" - Auto-correlating data \n");
  timer(0);
  auto_angular_cross_bf(nfull_dat,indices_dat,pixrad_dat,DD);
  timer(2);
  print_info(" - Auto-correlating random \n");
  auto_angular_cross_bf(nfull_ran,indices_ran,pixrad_ran,RR);
  timer(2);
  print_info(" - Cross-correlating \n");
  cross_angular_cross_bf(nfull_dat,indices_dat,
      pixrad_dat,pixrad_ran,DR);
  timer(1);

  print_info("\n");
  write_CF(fnameOut,DD,DR,RR,
      sum_wd,sum_wd2,sum_wr,sum_wr2);

  print_info("*** Cleaning up\n");
  free_RadialPixels(n_boxes2D,pixrad_dat);
  free_RadialPixels(n_boxes2D,pixrad_ran);
  free(indices_dat);
  free(indices_ran);
  free(DD);
  free(DR);
  free(RR);
}

void run_angular_cross_corr_pm(void)
{
  //////
  // Runs w(theta) in PM mode
  np_t sum_wd,sum_wd2,sum_wr,sum_wr2;
  Catalog *cat_dat,*cat_ran;

  Cell2D *cells_dat,*cells_ran,*cells_dat_total,*cells_ran_total;
  int *indices_dat,*indices_ran;
  int nfull_dat,nfull_ran;

  histo_t *DD=(histo_t *)my_calloc(nb_red*(nb_red+1)*nb_theta/2,sizeof(histo_t));
  histo_t *DR=(histo_t *)my_calloc(nb_red*(nb_red+1)*nb_theta/2,sizeof(histo_t));
  histo_t *RR=(histo_t *)my_calloc(nb_red*(nb_red+1)*nb_theta/2,sizeof(histo_t));

  timer(4);

#ifdef _VERBOSE
  print_info("*** Angular cross-correlations: \n");
  print_info(" - Redshift range: %.3lf < z_mean < %.3lf\n",
      red_0,red_0+1./i_red_interval);
  print_info(" - # redshift bins: %d\n",nb_red);
  print_info(" - Redshift bin width: %.3lf\n",1/(i_red_interval*nb_red));

  print_info(" - Angular range: %.3lf < theta < %.3lf \n",
      0.,1./(i_theta_max*DTORAD));
  print_info(" - # angular bins : %d\n",nb_theta);
  if(logbin) {
    print_info(" - Logarithmic binning with %d bins per decade\n",
        n_logint);
  }
  else {
    print_info(" - Resolution: D(theta) = %.3lf \n",
        1./(i_theta_max*nb_theta*DTORAD));
  }

  print_info(" - Using a PM approach \n");
  print_info("\n");
#endif

  read_dr_catalogs(&cat_dat,&cat_ran,
      &sum_wd,&sum_wd2,&sum_wr,&sum_wr2);

  print_info("*** Boxing catalogs \n");
  init_2D_params(cat_dat,cat_ran,1);
  cells_dat=mk_Cells2D_many_from_Catalog(cat_dat,&indices_dat,&cells_dat_total,&nfull_dat);
#ifdef _CUTE_AS_PYTHON_MODULE
  if(global_galaxy_catalog == NULL)
#endif
    free_Catalog(cat_dat);
  free(indices_dat);
  cells_ran=mk_Cells2D_many_from_Catalog(cat_ran,&indices_ran,&cells_ran_total,&nfull_ran);
#ifdef _CUTE_AS_PYTHON_MODULE
  if(global_random_catalog == NULL)
#endif
    free_Catalog(cat_ran);
  free(indices_ran);
  print_info("\n");

#ifdef _DEBUG
  write_Cells2D(n_boxes2D,cells_dat_total,"debug_Cell2DDat.dat");
  write_Cells2D(n_boxes2D,cells_ran_total,"debug_Cell2DRan.dat");
#endif //_DEBUG

  print_info("*** Correlating \n");
  timer(0);
  corr_angular_cross_pm(cells_dat,cells_dat_total,
      cells_ran,cells_ran_total,
      DD,DR,RR);
  timer(1);

  print_info("\n");
  write_CF(fnameOut,DD,DR,RR,
      sum_wd,sum_wd2,sum_wr,sum_wr2);

  print_info("*** Cleaning up\n");
  free_Cells2D(nb_red*n_boxes2D,cells_dat);
  free_Cells2D(nb_red*n_boxes2D,cells_ran);
  free_Cells2D(n_boxes2D,cells_dat_total);
  free_Cells2D(n_boxes2D,cells_ran_total);
  free(DD);
  free(DR);
  free(RR);
}

void run_full_corr_bf(void)
{
  //////
  // Runs xi(theta,dz,z) in brute-force mode
  np_t sum_wd,sum_wd2,sum_wr,sum_wr2;
  Catalog *cat_dat,*cat_ran;

  RadialPixel *pixrad_dat,*pixrad_ran;
  int *indices_dat,*indices_ran;
  int nfull_dat,nfull_ran;

  histo_t *DD=(histo_t *)my_calloc(nb_red*nb_dz*nb_theta,sizeof(histo_t));
  histo_t *DR=(histo_t *)my_calloc(nb_red*nb_dz*nb_theta,sizeof(histo_t));
  histo_t *RR=(histo_t *)my_calloc(nb_red*nb_dz*nb_theta,sizeof(histo_t));

  timer(4);

#ifdef _VERBOSE
  print_info("*** Full correlation function: \n");
  print_info(" - Redshift range: %.3lf < z_mean < %.3lf\n",
      red_0,red_0+1./i_red_interval);
  print_info(" - # redshift bins: %d\n",nb_red);
  print_info(" - Redshift bin width: %.3lf\n",1/(i_red_interval*nb_red));

  print_info(" - Angular range: %.3lf < theta < %.3lf \n",
      0.,1./(i_theta_max*DTORAD));
  print_info(" - # angular bins : %d\n",nb_theta);
  if(logbin) {
    print_info(" - Logarithmic binning with %d bins per decade\n",
        n_logint);
  }
  else {
    print_info(" - Resolution: D(theta) = %.3lf \n",
        1./(i_theta_max*nb_theta*DTORAD));
  }

  print_info(" - Radial range: %.3lf < Dz < %.3lf \n",
      0.,1/i_dz_max);
  print_info(" - # radial bins : %d\n",nb_dz);
  print_info(" - Radial resolution: D(Dz) = %.3lf \n",
      1./(i_dz_max*nb_dz));

  print_info(" - Using a brute-force approach \n");
  print_info("\n");
#endif

  read_dr_catalogs(&cat_dat,&cat_ran,
      &sum_wd,&sum_wd2,&sum_wr,&sum_wr2);

  print_info("*** Boxing catalogs \n");
  init_2D_params(cat_dat,cat_ran,5);
  pixrad_dat=mk_RadialPixels_from_Catalog(cat_dat,&indices_dat,&nfull_dat,5);
#ifdef _CUTE_AS_PYTHON_MODULE
  if(global_galaxy_catalog == NULL)
#endif
    free_Catalog(cat_dat);
  pixrad_ran=mk_RadialPixels_from_Catalog(cat_ran,&indices_ran,&nfull_ran,5);
#ifdef _CUTE_AS_PYTHON_MODULE
  if(global_random_catalog == NULL)
#endif
    free_Catalog(cat_ran);
  print_info("\n");

#ifdef _DEBUG
  write_PixRads(n_boxes2D,pixrad_dat,"debug_PixRadDat.dat");
  write_PixRads(n_boxes2D,pixrad_ran,"debug_PixRadRan.dat");
#endif //_DEBUG

  print_info("*** Correlating \n");
  print_info(" - Auto-correlating data \n");
  timer(0);
  auto_full_bf(nfull_dat,indices_dat,pixrad_dat,DD);
  timer(2);
  print_info(" - Auto-correlating random \n");
  auto_full_bf(nfull_ran,indices_ran,pixrad_ran,RR);
  timer(2);
  print_info(" - Cross-correlating \n");
  cross_full_bf(nfull_dat,indices_dat,
      pixrad_dat,pixrad_ran,DR);
  timer(1);

  print_info("\n");
  write_CF(fnameOut,DD,DR,RR,
      sum_wd,sum_wd2,sum_wr,sum_wr2);

  print_info("*** Cleaning up\n");
  free_RadialPixels(n_boxes2D,pixrad_dat);
  free_RadialPixels(n_boxes2D,pixrad_ran);
  free(indices_dat);
  free(indices_ran);
  free(DD);
  free(DR);
  free(RR);
}

void run_full_corr_pm(void)
{
  //////
  // Runs xi(theta,dz,z) in PM mode
  np_t sum_wd,sum_wd2,sum_wr,sum_wr2;
  Catalog *cat_dat,*cat_ran;

  RadialCell *radcell_dat,*radcell_ran;

  histo_t *DD=(histo_t *)my_calloc(nb_red*nb_dz*nb_theta,sizeof(histo_t));
  histo_t *DR=(histo_t *)my_calloc(nb_red*nb_dz*nb_theta,sizeof(histo_t));
  histo_t *RR=(histo_t *)my_calloc(nb_red*nb_dz*nb_theta,sizeof(histo_t));

  timer(4);

#ifdef _VERBOSE
  print_info("*** Full correlation function: \n");
  print_info(" - Redshift range: %.3lf < z_mean < %.3lf\n",
      red_0,red_0+1./i_red_interval);
  print_info(" - # redshift bins: %d\n",nb_red);
  print_info(" - Redshift bin width: %.3lf\n",1/(i_red_interval*nb_red));

  print_info(" - Angular range: %.3lf < theta < %.3lf \n",
      0.,1./(i_theta_max*DTORAD));
  print_info(" - # angular bins : %d\n",nb_theta);
  if(logbin) {
    print_info(" - Logarithmic binning with %d bins per decade\n",
        n_logint);
  }
  else {
    print_info(" - Resolution: D(theta) = %.3lf \n",
        1./(i_theta_max*nb_theta*DTORAD));
  }

  print_info(" - Radial range: %.3lf < Dz < %.3lf \n",
      0.,1/i_dz_max);
  print_info(" - # radial bins : %d\n",nb_dz);
  print_info(" - Radial resolution: D(Dz) = %.3lf \n",
      1./(i_dz_max*nb_dz));

  print_info(" - Using a PM approach \n");
  print_info("\n");
#endif

  read_dr_catalogs(&cat_dat,&cat_ran,
      &sum_wd,&sum_wd2,&sum_wr,&sum_wr2);

  print_info("*** Boxing catalogs \n");
  init_2D_params(cat_dat,cat_ran,5);
  radcell_dat=mk_RadialCells_from_Catalog(cat_dat);
#ifdef _CUTE_AS_PYTHON_MODULE
  if(global_galaxy_catalog == NULL)
#endif
    free_Catalog(cat_dat);
  radcell_ran=mk_RadialCells_from_Catalog(cat_ran);
#ifdef _CUTE_AS_PYTHON_MODULE
  if(global_random_catalog == NULL)
#endif
    free_Catalog(cat_ran);
  print_info("\n");

  print_info("*** Correlating \n");
  timer(0);
  corr_full_pm(radcell_dat,radcell_ran,DD,DR,RR);
  timer(1);

  print_info("\n");
  write_CF(fnameOut,DD,DR,RR,
      sum_wd,sum_wd2,sum_wr,sum_wr2);

  print_info("*** Cleaning up\n");
  free_RadialCells(n_boxes2D,radcell_dat);
  free_RadialCells(n_boxes2D,radcell_ran);
  free(DD);
  free(DR);
  free(RR);
}

void run_radial_corr_bf(void)
{
  //////
  // Runs xi(dz,alpha) in brute-force mode
  np_t sum_wd,sum_wd2,sum_wr,sum_wr2;
  Catalog *cat_dat,*cat_ran;

  RadialPixel *pixrad_dat,*pixrad_ran;
  int *indices_dat,*indices_ran;
  int nfull_dat,nfull_ran;

  histo_t *DD=(histo_t *)my_calloc(nb_dz,sizeof(histo_t));
  histo_t *DR=(histo_t *)my_calloc(nb_dz,sizeof(histo_t));
  histo_t *RR=(histo_t *)my_calloc(nb_dz,sizeof(histo_t));

  timer(4);

#ifdef _VERBOSE
  print_info("*** Radial correlation function: \n");
  print_info(" - Range: %.3lf < Dz < %.3lf \n",
      0.,1/i_dz_max);
  print_info(" - #bins: %d\n",nb_dz);
  print_info(" - Resolution: D(Dz) = %.3lf \n",
      1./(i_dz_max*nb_dz));
  print_info(" - Using a brute-force approach \n");
  print_info(" - Colinear galaxies within Dtheta = %.3lf (deg) \n",
      aperture_los/DTORAD);
  print_info("\n");
#endif

  read_dr_catalogs(&cat_dat,&cat_ran,
      &sum_wd,&sum_wd2,&sum_wr,&sum_wr2);

  print_info("*** Boxing catalogs \n");
  init_2D_params(cat_dat,cat_ran,0);
  pixrad_dat=mk_RadialPixels_from_Catalog(cat_dat,&indices_dat,&nfull_dat,0);
#ifdef _CUTE_AS_PYTHON_MODULE
  if(global_galaxy_catalog == NULL)
#endif
    free_Catalog(cat_dat);
  pixrad_ran=mk_RadialPixels_from_Catalog(cat_ran,&indices_ran,&nfull_ran,0);
#ifdef _CUTE_AS_PYTHON_MODULE
  if(global_random_catalog == NULL)
#endif
    free_Catalog(cat_ran);
  print_info("\n");

#ifdef _DEBUG
  write_PixRads(n_boxes2D,pixrad_dat,"debug_PixRadDat.dat");
  write_PixRads(n_boxes2D,pixrad_ran,"debug_PixRadRan.dat");
#endif //_DEBUG

  print_info("*** Correlating \n");
  print_info(" - Auto-correlating data \n");
  timer(0);
  auto_rad_bf(nfull_dat,indices_dat,pixrad_dat,DD);
  timer(2);
  print_info(" - Auto-correlating random \n");
  auto_rad_bf(nfull_ran,indices_ran,pixrad_ran,RR);
  timer(2);
  print_info(" - Cross-correlating \n");
  cross_rad_bf(nfull_dat,indices_dat,
      pixrad_dat,pixrad_ran,DR);
  timer(1);

  print_info("\n");
  write_CF(fnameOut,DD,DR,RR,
      sum_wd,sum_wd2,sum_wr,sum_wr2);

  print_info("*** Cleaning up\n");
  free_RadialPixels(n_boxes2D,pixrad_dat);
  free_RadialPixels(n_boxes2D,pixrad_ran);
  free(indices_dat);
  free(indices_ran);
  free(DD);
  free(DR);
  free(RR);
}

void run_angular_corr_bf(void)
{
  //////
  // Runs w(theta) in brute-force mode
  np_t sum_wd,sum_wd2,sum_wr,sum_wr2;
  Catalog *cat_dat,*cat_ran;

  Box2D *boxes_dat,*boxes_ran;
  int *indices_dat,*indices_ran;
  int nfull_dat,nfull_ran;

  histo_t *DD=(histo_t *)my_calloc(nb_theta,sizeof(histo_t));
  histo_t *DR=(histo_t *)my_calloc(nb_theta,sizeof(histo_t));
  histo_t *RR=(histo_t *)my_calloc(nb_theta,sizeof(histo_t));

  timer(4);

#ifdef _VERBOSE
  print_info("*** Angular correlation function: \n");
  print_info(" - Range: %.3lf < theta < %.3lf (deg)\n",
      0.,1/(i_theta_max*DTORAD));
  print_info(" - #bins: %d\n",nb_theta);
  if(logbin) {
    print_info(" - Logarithmic binning with %d bins per decade\n",
        n_logint);
  }
  else {
    print_info(" - Resolution: D(theta) = %.3lf \n",
        1./(i_theta_max*nb_theta*DTORAD));
  }
  print_info(" - Using a brute-force approach \n");
  print_info("\n");
#endif

  read_dr_catalogs(&cat_dat,&cat_ran,
      &sum_wd,&sum_wd2,&sum_wr,&sum_wr2);

  print_info("*** Boxing catalogs \n");
  init_2D_params(cat_dat,cat_ran,1);
  boxes_dat=mk_Boxes2D_from_Catalog(cat_dat,&indices_dat,&nfull_dat);
#ifdef _CUTE_AS_PYTHON_MODULE
  if(global_galaxy_catalog == NULL)
#endif
    free_Catalog(cat_dat);
  boxes_ran=mk_Boxes2D_from_Catalog(cat_ran,&indices_ran,&nfull_ran);
#ifdef _CUTE_AS_PYTHON_MODULE
  if(global_random_catalog == NULL)
#endif
    free_Catalog(cat_ran);
  print_info("\n");

#ifdef _DEBUG
  write_Boxes2D(n_boxes2D,boxes_dat,"debug_Box2DDat.dat");
  write_Boxes2D(n_boxes2D,boxes_ran,"debug_Box2DRan.dat");
#endif //_DEBUG

  print_info("*** Correlating \n");
  print_info(" - Auto-correlating data \n");
  timer(0);
  auto_ang_bf(nfull_dat,indices_dat,boxes_dat,DD);
  timer(2);
  print_info(" - Auto-correlating random \n");
  auto_ang_bf(nfull_ran,indices_ran,boxes_ran,RR);
  timer(2);
  print_info(" - Cross-correlating \n");
  cross_ang_bf(nfull_dat,indices_dat,
      boxes_dat,boxes_ran,DR);
  timer(1);

  print_info("\n");
  write_CF(fnameOut,DD,DR,RR,
      sum_wd,sum_wd2,sum_wr,sum_wr2);

  print_info("*** Cleaning up\n");
  free_Boxes2D(n_boxes2D,boxes_dat);
  free_Boxes2D(n_boxes2D,boxes_ran);
  free(indices_dat);
  free(indices_ran);
  free(DD);
  free(DR);
  free(RR);
}

void run_angular_corr_pm(void)
{
  //////
  // Runs w(theta) in PM mode
  np_t sum_wd,sum_wd2,sum_wr,sum_wr2;
  Catalog *cat_dat,*cat_ran;

  Cell2D *cells_dat,*cells_ran;
  int *indices_dat,*indices_ran;
  int nfull_dat,nfull_ran;

  histo_t *DD=(histo_t *)my_calloc(nb_theta,sizeof(histo_t));
  histo_t *DR=(histo_t *)my_calloc(nb_theta,sizeof(histo_t));
  histo_t *RR=(histo_t *)my_calloc(nb_theta,sizeof(histo_t));

  timer(4);

#ifdef _VERBOSE
  print_info("*** Angular correlation function: \n");
  print_info(" - Range: %.3lf < theta < %.3lf (deg)\n",
      0.,1/(i_theta_max*DTORAD));
  print_info(" - #bins: %d\n",nb_theta);
  if(logbin) {
    print_info(" - Logarithmic binning with %d bins per decade\n",
        n_logint);
  }
  else {
    print_info(" - Resolution: D(theta) = %.3lf \n",
        1./(i_theta_max*nb_theta*DTORAD));
  }
  print_info(" - Using a PM approach \n");
  print_info("\n");
#endif

  read_dr_catalogs(&cat_dat,&cat_ran,
      &sum_wd,&sum_wd2,&sum_wr,&sum_wr2);

  print_info("*** Boxing catalogs \n");
  init_2D_params(cat_dat,cat_ran,1);
  cells_dat=mk_Cells2D_from_Catalog(cat_dat,&indices_dat,&nfull_dat);
#ifdef _CUTE_AS_PYTHON_MODULE
  if(global_galaxy_catalog == NULL)
#endif
    free_Catalog(cat_dat);
  free(indices_dat);
  cells_ran=mk_Cells2D_from_Catalog(cat_ran,&indices_ran,&nfull_ran);
#ifdef _CUTE_AS_PYTHON_MODULE
  if(global_random_catalog == NULL)
#endif
    free_Catalog(cat_ran);
  free(indices_ran);
  print_info("\n");

#ifdef _DEBUG
  write_Cells2D(n_boxes2D,cells_dat,"debug_Cell2DDat.dat");
  write_Cells2D(n_boxes2D,cells_ran,"debug_Cell2DRan.dat");
#endif //_DEBUG

  print_info("*** Correlating \n");
  timer(0);
  corr_ang_pm(cells_dat,cells_ran,DD,DR,RR);
  timer(1);

  print_info("\n");
  write_CF(fnameOut,DD,DR,RR,
      sum_wd,sum_wd2,sum_wr,sum_wr2);

  print_info("*** Cleaning up\n");
  free_Cells2D(n_boxes2D,cells_dat);
  free_Cells2D(n_boxes2D,cells_ran);
  free(DD);
  free(DR);
  free(RR);
}

void run_monopole_corr_bf(void)
{
  //////
  // Runs xi(r) in brute-force mode
  np_t sum_wd,sum_wd2,sum_wr,sum_wr2;
  Catalog *cat_dat,*cat_ran;

  Box3D *boxes_dat,*boxes_ran;
  int *indices_dat,*indices_ran;
  int nfull_dat,nfull_ran;

  histo_t *DD=(histo_t *)my_calloc(nb_r,sizeof(histo_t));
  histo_t *DR=(histo_t *)my_calloc(nb_r,sizeof(histo_t));
  histo_t *RR=(histo_t *)my_calloc(nb_r,sizeof(histo_t));

  timer(4);

  set_r_z();

#ifdef _VERBOSE
  print_info("*** Monopole correlation function: \n");
  print_info(" - Range: %.3lf < r < %.3lf Mpc/h\n",
      0.,1/i_r_max);
  print_info(" - #bins: %d\n",nb_r);
  if(logbin) {
    print_info(" - Logarithmic binning with %d bins per decade\n",
        n_logint);
  }
  else {
    print_info(" - Resolution: D(r) = %.3lf Mpc/h\n",
        1./(i_r_max*nb_r));
  }
  print_info(" - Using a brute-force approach \n");
  print_info("\n");
#endif

  read_dr_catalogs(&cat_dat,&cat_ran,
      &sum_wd,&sum_wd2,&sum_wr,&sum_wr2);

  print_info("*** Boxing catalogs \n");
  init_3D_params(cat_dat,cat_ran,2);
  boxes_dat=mk_Boxes3D_from_Catalog(cat_dat,&indices_dat,&nfull_dat);
#ifdef _CUTE_AS_PYTHON_MODULE
  if(global_galaxy_catalog == NULL)
#endif
    free_Catalog(cat_dat);
  boxes_ran=mk_Boxes3D_from_Catalog(cat_ran,&indices_ran,&nfull_ran);
#ifdef _CUTE_AS_PYTHON_MODULE
  if(global_random_catalog == NULL)
#endif
    free_Catalog(cat_ran);
  print_info("\n");

#ifdef _DEBUG
  write_Boxes3D(n_boxes3D,boxes_dat,"debug_Box3DDat.dat");
  write_Boxes3D(n_boxes3D,boxes_ran,"debug_Box3DRan.dat");
#endif //_DEBUG

  print_info("*** Correlating \n");
  print_info(" - Auto-correlating data \n");
  timer(0);
  auto_mono_bf(nfull_dat,indices_dat,boxes_dat,DD);
  timer(2);
  print_info(" - Auto-correlating random \n");
  auto_mono_bf(nfull_ran,indices_ran,boxes_ran,RR);
  timer(2);
  print_info(" - Cross-correlating \n");
  cross_mono_bf(nfull_dat,indices_dat,
      boxes_dat,boxes_ran,DR);
  timer(1);

  print_info("\n");
  write_CF(fnameOut,DD,DR,RR,
      sum_wd,sum_wd2,sum_wr,sum_wr2);

  print_info("*** Cleaning up\n");
  free_Boxes3D(n_boxes3D,boxes_dat);
  free_Boxes3D(n_boxes3D,boxes_ran);
  free(indices_dat);
  free(indices_ran);
  end_r_z();
  free(DD);
  free(DR);
  free(RR);
}



void run_3d_ps_corr_bf(void)
{
  //////
  // Runs xi(pi,sigma) in brute-force mode
  np_t sum_wd,sum_wd2,sum_wr,sum_wr2;
  Catalog *cat_dat,*cat_ran;

  Box3D *boxes_dat,*boxes_ran;
  int *indices_dat,*indices_ran;
  int nfull_dat,nfull_ran;

  histo_t *DD=(histo_t *)my_calloc(nb_rt*nb_rl,sizeof(histo_t));
  histo_t *DR=(histo_t *)my_calloc(nb_rt*nb_rl,sizeof(histo_t));
  histo_t *RR=(histo_t *)my_calloc(nb_rt*nb_rl,sizeof(histo_t));

  timer(4);

  set_r_z();

#ifdef _VERBOSE
  print_info("*** 3D correlation function (pi,sigma): \n");
  print_info(" - Range: (%.3lf,%.3lf) < (pi,sigma) < (%.3lf,%.3lf) Mpc/h\n",
      0.,0.,1/i_rl_max,1/i_rt_max);
  print_info(" - #bins: (%d,%d)\n",nb_rl,nb_rt);
  print_info(" - Resolution: (d(pi),d(sigma)) = (%.3lf,%.3lf) Mpc/h\n",
      1./(i_rl_max*nb_rl),1./(i_rt_max*nb_rt));
  print_info(" - Using a brute-force approach \n");
  print_info("\n");
#endif

  read_dr_catalogs(&cat_dat,&cat_ran,
      &sum_wd,&sum_wd2,&sum_wr,&sum_wr2);

  print_info("*** Boxing catalogs \n");
  init_3D_params(cat_dat,cat_ran,3);
  boxes_dat=mk_Boxes3D_from_Catalog(cat_dat,&indices_dat,&nfull_dat);
#ifdef _CUTE_AS_PYTHON_MODULE
  if(global_galaxy_catalog == NULL)
#endif
    free_Catalog(cat_dat);
  boxes_ran=mk_Boxes3D_from_Catalog(cat_ran,&indices_ran,&nfull_ran);
#ifdef _CUTE_AS_PYTHON_MODULE
  if(global_random_catalog == NULL)
#endif
    free_Catalog(cat_ran);
  print_info("\n");

#ifdef _DEBUG
  write_Boxes3D(n_boxes3D,boxes_dat,"debug_Box3DDat.dat");
  write_Boxes3D(n_boxes3D,boxes_ran,"debug_Box3DRan.dat");
#endif //_DEBUG

  print_info("*** Correlating \n");
  print_info(" - Auto-correlating data \n");
  timer(0);
  auto_3d_ps_bf(nfull_dat,indices_dat,boxes_dat,DD);
  timer(2);
  print_info(" - Auto-correlating random \n");
  auto_3d_ps_bf(nfull_ran,indices_ran,boxes_ran,RR);
  timer(2);
  print_info(" - Cross-correlating \n");
  cross_3d_ps_bf(nfull_dat,indices_dat,
      boxes_dat,boxes_ran,DR);
  timer(1);

  print_info("\n");
  write_CF(fnameOut,DD,DR,RR,
      sum_wd,sum_wd2,sum_wr,sum_wr2);

  print_info("*** Cleaning up\n");
  free_Boxes3D(n_boxes3D,boxes_dat);
  free_Boxes3D(n_boxes3D,boxes_ran);
  free(indices_dat);
  free(indices_ran);
  end_r_z();
  free(DD);
  free(DR);
  free(RR);
}

void run_3d_rm_corr_bf(void)
{
  //////
  // Runs xi(r,mu) in brute-force mode
  np_t sum_wd,sum_wd2,sum_wr,sum_wr2;
  Catalog *cat_dat,*cat_ran;

  Box3D *boxes_dat,*boxes_ran;
  int *indices_dat,*indices_ran;
  int nfull_dat,nfull_ran;

  histo_t *DD=(histo_t *)my_calloc(nb_r*nb_mu,sizeof(histo_t));
  histo_t *DR=(histo_t *)my_calloc(nb_r*nb_mu,sizeof(histo_t));
  histo_t *RR=(histo_t *)my_calloc(nb_r*nb_mu,sizeof(histo_t));

  timer(4);

  set_r_z();

#ifdef _VERBOSE
  print_info("*** 3D correlation function (r,mu): \n");
  print_info(" - Range: %.3lf < r < %.3lf Mpc/h\n",
      0.,1/i_r_max);
  print_info(" - #bins: %d\n",nb_r);
  print_info(" - Range: 0.000 < mu < 1.000\n");
  print_info(" - #bins: %d\n",nb_mu);
  if(logbin) {
    print_info(" - Logarithmic binning with %d bins per decade\n",
        n_logint);
  }
  else {
    print_info(" - Resolution: d(r) = %.3lf Mpc/h\n",
        1./(i_r_max*nb_r));
  }
  print_info(" - Using a brute-force approach \n");
  print_info("\n");
#endif

  read_dr_catalogs(&cat_dat,&cat_ran,
      &sum_wd,&sum_wd2,&sum_wr,&sum_wr2);

  print_info("*** Boxing catalogs \n");
  init_3D_params(cat_dat,cat_ran,4);
  boxes_dat=mk_Boxes3D_from_Catalog(cat_dat,&indices_dat,&nfull_dat);
#ifdef _CUTE_AS_PYTHON_MODULE
  if(global_galaxy_catalog == NULL)
#endif
    free_Catalog(cat_dat);
  boxes_ran=mk_Boxes3D_from_Catalog(cat_ran,&indices_ran,&nfull_ran);
#ifdef _CUTE_AS_PYTHON_MODULE
  if(global_random_catalog == NULL)
#endif
    free_Catalog(cat_ran);
  print_info("\n");

#ifdef _DEBUG
  write_Boxes3D(n_boxes3D,boxes_dat,"debug_Box3DDat.dat");
  write_Boxes3D(n_boxes3D,boxes_ran,"debug_Box3DRan.dat");
#endif //_DEBUG

  print_info("*** Correlating \n");
  print_info(" - Auto-correlating data \n");
  timer(0);
  auto_3d_rm_bf(nfull_dat,indices_dat,boxes_dat,DD);
  timer(2);
  print_info(" - Auto-correlating random \n");
  auto_3d_rm_bf(nfull_ran,indices_ran,boxes_ran,RR);
  timer(2);
  print_info(" - Cross-correlating \n");
  cross_3d_rm_bf(nfull_dat,indices_dat,
      boxes_dat,boxes_ran,DR);
  timer(1);

  print_info("\n");
  write_CF(fnameOut,DD,DR,RR,
      sum_wd,sum_wd2,sum_wr,sum_wr2);

  print_info("*** Cleaning up\n");
  free_Boxes3D(n_boxes3D,boxes_dat);
  free_Boxes3D(n_boxes3D,boxes_ran);
  free(indices_dat);
  free(indices_ran);
  end_r_z();
  free(DD);
  free(DR);
  free(RR);
}

void run_monopole_cross_corr_bf(int reuse_ran)
{
  //////
  // Runs xi(r) in brute-force mode for cross-correlation between 2 tracers
  np_t sum_wd1,sum_wd2,sum_wr1,sum_wr2,junk;
  Catalog *cat_dat1,*cat_dat2,*cat_ran1,*cat_ran2;

  Box3D *boxes_dat1,*boxes_dat2,*boxes_ran1,*boxes_ran2;
  int *indices_dat1,*indices_dat2,*indices_ran1,*indices_ran2;
  int nfull_dat1,nfull_dat2,nfull_ran1,nfull_ran2,ii;

  histo_t *D1D2=(histo_t *)my_calloc(nb_r,sizeof(histo_t));
  histo_t *D1R=(histo_t *)my_calloc(nb_r,sizeof(histo_t));
  histo_t *D2R=(histo_t *)my_calloc(nb_r,sizeof(histo_t));
  histo_t *RR=(histo_t *)my_calloc(nb_r,sizeof(histo_t));

  timer(4);

  set_r_z();

#ifdef _VERBOSE
  print_info("*** Monopole cross-correlation function: \n");
  print_info(" - Range: %.3lf < r < %.3lf Mpc/h\n",
      0.,1/i_r_max);
  print_info(" - #bins: %d\n",nb_r);
  if(logbin) {
    print_info(" - Logarithmic binning with %d bins per decade\n",
        n_logint);
  }
  else {
    print_info(" - Resolution: D(r) = %.3lf Mpc/h\n",
        1./(i_r_max*nb_r));
  }
  print_info(" - Using a brute-force approach \n");
  print_info("\n");
#endif

  read_ddrr_catalogs(&cat_dat1,&cat_dat2,&cat_ran1,&cat_ran2,
      &sum_wd1,&sum_wd2,&sum_wr1,&sum_wr2,&junk);

  print_info("*** Boxing catalogs \n");
  init_3D_params_cross(cat_dat1,cat_dat2,cat_ran1,cat_ran2,2);
  boxes_dat1=mk_Boxes3D_from_Catalog(cat_dat1,&indices_dat1,&nfull_dat1);
#ifdef _CUTE_AS_PYTHON_MODULE
  if(global_galaxy_catalog == NULL)
#endif
    free_Catalog(cat_dat1);
  boxes_dat2=mk_Boxes3D_from_Catalog(cat_dat2,&indices_dat2,&nfull_dat2);
#ifdef _CUTE_AS_PYTHON_MODULE
  if(global_galaxy_catalog2 == NULL)
#endif
    free_Catalog(cat_dat2);
  boxes_ran1=mk_Boxes3D_from_Catalog(cat_ran1,&indices_ran1,&nfull_ran1);
#ifdef _CUTE_AS_PYTHON_MODULE
  if(global_random_catalog == NULL)
#endif
    free_Catalog(cat_ran1);
  boxes_ran2=mk_Boxes3D_from_Catalog(cat_ran2,&indices_ran2,&nfull_ran2);
#ifdef _CUTE_AS_PYTHON_MODULE
  if(global_random_catalog2 == NULL)
#endif
    free_Catalog(cat_ran2);
  print_info("\n");

#ifdef _DEBUG
  write_Boxes3D(n_boxes3D,boxes_dat1,"debug_Box3DDat1.dat");
  write_Boxes3D(n_boxes3D,boxes_dat2,"debug_Box3DDat2.dat");
  write_Boxes3D(n_boxes3D,boxes_ran1,"debug_Box3DRan1.dat");
  write_Boxes3D(n_boxes3D,boxes_ran2,"debug_Box3DRan2.dat");
#endif //_DEBUG

  print_info("*** Correlating \n");
  print_info(" - Cross-correlating D1 and D2 \n");
  timer(0);
  cross_mono_bf(nfull_dat1,indices_dat1,
      boxes_dat1,boxes_dat2,D1D2);
  timer(2);
  print_info(" - Cross-correlating D1 and R2 \n");
  cross_mono_bf(nfull_dat1,indices_dat1,
      boxes_dat1,boxes_ran2,D1R);
  timer(2);
  print_info(" - Cross-correlating D2 and R1 \n");
  /*cross_mono_bf(nfull_dat2,indices_dat2,
    boxes_dat2,boxes_ran,D2R);*/
  cross_mono_bf(nfull_ran1,indices_ran1,
      boxes_ran1,boxes_dat2,D2R);
  timer(2);
  if(!reuse_ran) {
    print_info(" - Cross-correlating R1 and R2 \n");
    cross_mono_bf(nfull_ran1,indices_ran1,boxes_ran1,
        boxes_ran2,RR);
  }
  else {
    print_info(" - Skipping R1R2 correlation; look up from file and recalculate xi output later! \n");
    for(ii=0;ii<nb_r;ii++)
      RR[ii]=0;
  }
  timer(1);


  print_info("\n");
  write_CCF(fnameOut,D1D2,D1R,D2R,RR,
      sum_wd1,sum_wd2,sum_wr1,sum_wr2,reuse_ran);

  print_info("*** Cleaning up\n");
  free_Boxes3D(n_boxes3D,boxes_dat1);
  free_Boxes3D(n_boxes3D,boxes_dat2);
  free_Boxes3D(n_boxes3D,boxes_ran1);
  free_Boxes3D(n_boxes3D,boxes_ran2);
  free(indices_dat1);
  free(indices_dat2);
  free(indices_ran1);
  free(indices_ran2);
  end_r_z();
  free(D1D2);
  free(D1R);
  free(D2R);
  free(RR);
}

void run_3d_rm_cross_corr_bf(int reuse_ran)
{
  //////
  // Runs xi(r,mu) in brute-force mode for cross-correlation between 2 tracer populations
  np_t sum_wd1,sum_wd2,sum_wr1,sum_wr2,junk;
  Catalog *cat_dat1,*cat_dat2,*cat_ran1,*cat_ran2;

  Box3D *boxes_dat1,*boxes_dat2,*boxes_ran1,*boxes_ran2;
  int *indices_dat1,*indices_dat2,*indices_ran1,*indices_ran2;
  int nfull_dat1,nfull_dat2,nfull_ran1,nfull_ran2,ii;

  histo_t *D1D2=(histo_t *)my_calloc(nb_r*nb_mu,sizeof(histo_t));
  histo_t *D1R=(histo_t *)my_calloc(nb_r*nb_mu,sizeof(histo_t));
  histo_t *D2R=(histo_t *)my_calloc(nb_r*nb_mu,sizeof(histo_t));
  histo_t *RR=(histo_t *)my_calloc(nb_r*nb_mu,sizeof(histo_t));

  timer(4);

  set_r_z();

#ifdef _VERBOSE
  print_info("*** 3D cross-correlation function (r,mu): \n");
  print_info(" - Range: %.3lf < r < %.3lf Mpc/h\n",
      0.,1/i_r_max);
  print_info(" - #bins: %d\n",nb_r);
  print_info(" - Range: 0.000 < mu < 1.000\n");
  print_info(" - #bins: %d\n",nb_mu);
  if(logbin) {
    print_info(" - Logarithmic binning with %d bins per decade\n", n_logint);
  }
  else {
    print_info(" - Resolution: d(r) = %.3lf Mpc/h\n",
        1./(i_r_max*nb_r));
  }
  print_info(" - Using a brute-force approach \n");
  print_info("\n");
#endif

  read_ddrr_catalogs(&cat_dat1,&cat_dat2,&cat_ran1,&cat_ran2,
      &sum_wd1,&sum_wd2,&sum_wr1,&sum_wr2,&junk);

  print_info("*** Boxing catalogs \n");
  init_3D_params_cross(cat_dat1,cat_dat2,cat_ran1,cat_ran2,4);  // assumes 2nd data catalogue is the more numerous one!
  boxes_dat1=mk_Boxes3D_from_Catalog(cat_dat1,&indices_dat1,&nfull_dat1);
#ifdef _CUTE_AS_PYTHON_MODULE
  if(global_galaxy_catalog == NULL)
#endif
    free_Catalog(cat_dat1);
  boxes_dat2=mk_Boxes3D_from_Catalog(cat_dat2,&indices_dat2,&nfull_dat2);
#ifdef _CUTE_AS_PYTHON_MODULE
  if(global_galaxy_catalog2 == NULL)
#endif
    free_Catalog(cat_dat2);
  boxes_ran1=mk_Boxes3D_from_Catalog(cat_ran1,&indices_ran1,&nfull_ran1);
#ifdef _CUTE_AS_PYTHON_MODULE
  if(global_random_catalog == NULL)
#endif
    free_Catalog(cat_ran1);
  boxes_ran2=mk_Boxes3D_from_Catalog(cat_ran2,&indices_ran2,&nfull_ran2);
#ifdef _CUTE_AS_PYTHON_MODULE
  if(global_random_catalog2 == NULL)
#endif
    free_Catalog(cat_ran2);
  print_info("\n");

#ifdef _DEBUG
  write_Boxes3D(n_boxes3D,boxes_dat1,"debug_Box3DDat1.dat");
  write_Boxes3D(n_boxes3D,boxes_dat2,"debug_Box3DDat2.dat");
  write_Boxes3D(n_boxes3D,boxes_ran1,"debug_Box3DRan1.dat");
  write_Boxes3D(n_boxes3D,boxes_ran2,"debug_Box3DRan2.dat");
#endif //_DEBUG

  print_info("*** Correlating \n");
  print_info(" - Cross-correlating D1 and D2 \n");
  timer(0);
  print_info("   (defining angles wrt l-o-s direction to void centre) \n");
  cross_3d_rm_special_bf(nfull_dat1,indices_dat1,
      boxes_dat1,boxes_dat2,D1D2);
  timer(2);
  print_info(" - Cross-correlating D1 and R2 \n");
  cross_3d_rm_special_bf(nfull_dat1,indices_dat1,
      boxes_dat1,boxes_ran2,D1R);
  timer(2);
  print_info(" - Cross-correlating D2 and R1 \n");
  /*cross_3d_rm_bf(nfull_dat2,indices_dat2,
    boxes_dat2,boxes_ran,D2R);*/
  cross_3d_rm_special_bf(nfull_ran1,indices_ran1,
      boxes_ran1,boxes_dat2,D2R);
  timer(2);
  if(!reuse_ran) {
    print_info(" - Cross-correlating R1 and R2 \n");
    cross_3d_rm_special_bf(nfull_ran1,indices_ran1,
        boxes_ran1,boxes_ran2,RR);
  }
  else {
    print_info(" - Skipping R1R2 correlation; look up from file and recalculate xi output later! \n");
    for(ii=0;ii<nb_r;ii++)
      RR[ii]=0;
  }
  timer(1);

  print_info("\n");
  write_CCF(fnameOut,D1D2,D1R,D2R,RR,
      sum_wd1,sum_wd2,sum_wr1,sum_wr2,reuse_ran);

  print_info("*** Cleaning up\n");
  free_Boxes3D(n_boxes3D,boxes_dat1);
  free_Boxes3D(n_boxes3D,boxes_dat2);
  free_Boxes3D(n_boxes3D,boxes_ran1);
  free_Boxes3D(n_boxes3D,boxes_ran2);
  free(indices_dat1);
  free(indices_dat2);
  free(indices_ran1);
  free(indices_ran2);
  end_r_z();
  free(D1D2);
  free(D1R);
  free(D2R);
  free(RR);
}

#ifdef _CUTE_AS_PYTHON_MODULE

int mpi_init_called = 0;
void finalize_mpi(){
#ifdef _HAVE_MPI
  MPI_Finalize();
#endif //_HAVE_MPI
}

int runCUTE(Catalog *galaxy_catalog, Catalog *galaxy_catalog2, Catalog *random_catalog, Catalog *random_catalog2, Result *result){

  int ii;

  // We can only call MPI_Init once
  if(mpi_init_called == 0){
    mpi_init(NULL,NULL);
    mpi_init_called = 1;
  }

#else

int main(int argc,char **argv){
  //////
  // Main routine
  int ii;
  char fnameIn[128];
  if(argc!=2) {
    print_info("Usage ./CUTE <input file>\n");
    exit(1);
  }
  sprintf(fnameIn,"%s",argv[1]);

  mpi_init(&argc,&argv);

#endif

  setbuf(stdout, NULL);

  print_info("\n");
  print_info("-----------------------------------------------------------\n");
  print_info("|| CUTE - Correlation Utilities and Two-point Estimation ||\n");
  print_info("-----------------------------------------------------------\n\n");

#ifdef _CUTE_AS_PYTHON_MODULE

  global_galaxy_catalog  = galaxy_catalog;
  global_galaxy_catalog2 = galaxy_catalog2;
  global_random_catalog  = random_catalog;
  global_random_catalog2 = random_catalog2;
  
  // Initialize
  if(galaxy_catalog != NULL){
    print_info("Using external data catalog with np = %d  w = %0.1f  w2 = %0.1f\n", 
        galaxy_catalog->np, galaxy_catalog->sum_w, galaxy_catalog->sum_w2);
  }
  if(galaxy_catalog2 != NULL){
    print_info("Using second external data catalog with np = %d  w = %0.1f  w2 = %0.1f\n", 
        galaxy_catalog2->np, galaxy_catalog2->sum_w, galaxy_catalog2->sum_w2);
  }
  if(random_catalog != NULL){
    print_info("Using external random catalog with np = %d  w = %0.1f  w2 = %0.1f\n", 
        random_catalog->np, random_catalog->sum_w, random_catalog->sum_w2);
  }
  if(random_catalog2 != NULL){
    global_random_catalog2 = random_catalog2;
    print_info("Using second external random catalog with np = %d  w = %0.1f  w2 = %0.1f\n", 
        random_catalog2->np, random_catalog2->sum_w, random_catalog2->sum_w2);
  }

  global_result = result;

#ifdef _HAVE_MPI
  print_info("Running MPI with %i tasks\n",NNodes);
#endif

#endif

  //Initialize random number generator
#ifdef _DEBUG
  srand(1234);
#else
  srand(time(NULL));
#endif
#ifdef _VERBOSE
  print_info("Initializing random number generator\n");
  print_info("First random number : %d \n",rand());
#endif

#ifdef _VERBOSE
  //Calculate number of threads
  ii=0;
#pragma omp parallel
  {
#pragma omp atomic
    ii++;
  }
  print_info("Using %d threads \n",ii);
#endif
  print_info("\n");

#ifndef _CUTE_AS_PYTHON_MODULE
  read_run_params(fnameIn);
#endif

  if(corr_type==0)
    run_radial_corr_bf();
  else if(corr_type==1) {
    if(use_pm==1)
      run_angular_corr_pm();
    else
      run_angular_corr_bf();
  }
  else if(corr_type==2)
    run_monopole_corr_bf();
  else if(corr_type==3)
    run_3d_ps_corr_bf();
  else if(corr_type==4)
    run_3d_rm_corr_bf();
  else if(corr_type==5) {
    if(use_pm==1) {
      run_full_corr_pm();
      // run_full_corr_bf();
    }
    else
      run_full_corr_bf();
  }
  else if(corr_type==6) {
    if(use_pm==1) {
      run_angular_cross_corr_pm();
    }
    else
      run_angular_cross_corr_bf();
  }
  else if(corr_type==7) {
    run_monopole_cross_corr_bf(reuse_ran);
  }
  else if(corr_type==8) {
    run_3d_rm_cross_corr_bf(reuse_ran);
  }
  else {
    fprintf(stderr,"CUTE: wrong correlation type.\n");
    exit(0);
  }
  print_info("             Done !!!             \n");

#ifndef _CUTE_AS_PYTHON_MODULE
#ifdef _HAVE_MPI
  MPI_Finalize();
#endif //_HAVE_MPI
#endif

  return 0;
}
