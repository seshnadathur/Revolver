#include <math.h>
#include <stdlib.h>
#include "voz.h"

/*------------------------------------------------------------------------------
  Find nb elements of Real array a having the largest value.
  Returns index iord of these elements, ordered so iord[0] corresponds
  to element a[iord[0]] having the largest value.
  If nb > na, then the last nb-na elements of iord are undefined.
  Elements of a that are equal are left in their original order.

  Courtesy Andrew Hamilton.
*/
void findrtop(float *a, int na, int *iord, int nb)
{
#undef	ORDER
#define	ORDER(ia, ja)	a[ia] > a[ja] || (a[ia] == a[ja] && ia < ja)

  int i, ia, ib, it, n;
  
  n = (na <= nb)? na : nb;
  if (n <= 0) return;
  
  /* heap first n elements, so smallest element is at top of heap */
  for (ib = (n >> 1); ib < n; ib++) {
    iord[ib] = ib;
  }
  for (ia = (n >> 1) - 1; ia >= 0; ia--) {
    i = ia;
    for (ib = (i << 1) + 1; ib < n; ib = (i << 1) + 1) {
      if (ib+1 < n) {
	if (ORDER(iord[ib], iord[ib+1])) ib++;
      }
      if (ORDER(ia, iord[ib])) {
	iord[i] = iord[ib];
	i = ib;
      } else {
	break;
      }
    }
    iord[i] = ia;
  }
  
  /* now compare rest of elements of array to heap */
  for (ia = n; ia < na; ia++) {
    /* if new element is greater than smallest, sift it into heap */
    i = 0;
    if (ORDER(ia, iord[i])) {
      for (ib = (i << 1) + 1; ib < n; ib = (i << 1) + 1) {
	if (ib+1 < n) {
	  if (ORDER(iord[ib], iord[ib+1])) ib++;
	}
	if (ORDER(ia, iord[ib])) {
	  iord[i] = iord[ib];
	  i = ib;
	} else {
	  break;
	}
      }
      iord[i] = ia;
    }
  }
  
  /* unheap iord so largest element is at top */
  for (ia = n - 1; ia > 0; ia--) {
    it = iord[ia];
    i = 0;
    iord[ia] = iord[i];
    for (ib = (i << 1) + 1; ib < ia; ib = (i << 1) + 1) {
      if (ib+1 < ia) {
	if (ORDER(iord[ib], iord[ib+1])) ib++;
      }
      if (ORDER(it, iord[ib])) {
	iord[i] = iord[ib];
	i = ib;
      } else {
	break;
      }
    }
    iord[i] = it;
  }
}
