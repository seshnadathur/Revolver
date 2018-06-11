/* 
   Modified to match the precision of Qhull.
   All the VOBOZ files now include this file,
   and this one includes the necessary Qhull
   header file.

   REALfloat controls the precision in qhull/src/user.h.
   0 = double
   1 = single

   Rick Wagner - 15JAN08

*/

/* qhull_a.h includes qhull/src/user.h */
#include "qhull_a.h"

#define MAXVERVER 10000
#define NGUARD 42 /* Actually, the number of SPACES between guard points
		    in each dim */

/* number of particle to read in per chunk */
/* 1024**2 */
#define N_CHUNK 1048576

/* 
   These are needed for scanning strings correctly.
*/

#if (REALfloat == 1)
#define vozRealSym "f"
#elif (REALfloat == 0)
#define vozRealSym "lf"
#endif

/* REALmax is also from Qhull, and is precision dependent. */
#define BF REALmax

typedef struct Partadj {
  int nadj;
  int *adj;
} PARTADJ;


/* dimension loop */
#define DL for (d=0;d<3;d++)
