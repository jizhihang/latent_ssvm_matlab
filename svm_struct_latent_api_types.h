/************************************************************************/
/*                                                                      */
/*   svm_struct_latent_api_types.h                                      */
/*                                                                      */
/*   API type definitions for Latent SVM^struct                         */
/*                                                                      */
/*   Author: Chun-Nam Yu                                                */
/*   Date: 30.Sep.08                                                    */
/*                                                                      */
/*   This software is available for non-commercial use only. It must    */
/*   not be modified and distributed without prior permission of the    */
/*   author. The author is not responsible for implications from the    */
/*   use of this software.                                              */
/*                                                                      */
/************************************************************************/

# include "svm_light/svm_common.h"
#include <mex.h>


struct MexPhiCustomImpl_
{
  int counter ;
  mxArray * x ;
  mxArray * y ;
} ;

typedef struct MexPhiCustomImpl_ * MexPhiCustom ;

inline static MexPhiCustom
newMexPhiCustomFromPatternLabel (mxArray const * x, mxArray const *y)
{
  MexPhiCustom phi ;
  phi = mxMalloc (sizeof(struct MexPhiCustomImpl_)) ;
  phi -> counter = 1 ;
  phi -> x = mxDuplicateArray (x) ;
  phi -> y = mxDuplicateArray (y) ;
  return phi ;
}

inline static void
releaseMexPhiCustom (MexPhiCustom phi)
{
  if (phi) {
    phi -> counter -- ;
    if (phi -> counter == 0) {
      mxDestroyArray (phi -> x) ;
      mxDestroyArray (phi -> y) ;
      mxFree (phi) ;
    }
  }
}

inline static void
retainMexPhiCustom (MexPhiCustom phi) {
  if (phi) {
    phi -> counter ++ ;
  }
}

inline static mxArray *
MexPhiCustomGetPattern (MexPhiCustom phi) {
  assert (phi) ;
  return phi -> x ;
}

inline static mxArray *
MexPhiCustomGetLabel (MexPhiCustom phi) {
  assert (phi) ;
  return phi -> y ;
}


typedef struct MexKernelInfo_
{
  mxArray const * structParm ;
  mxArray const * kernelFn ;
} MexKernelInfo ;

inline static int
uIsString(const mxArray* A, int L)
{
  int M = mxGetM(A) ;
  int N = mxGetN(A) ;
  return
    mxIsChar(A) &&
    mxGetNumberOfDimensions(A) == 2 &&
    (M == 1 || (M == 0 && N == 0)) &&
    (L < 0 || N == L) ;
}

inline static int
uIsReal (const mxArray* A)
{
  return
    mxIsDouble(A) &&
    ! mxIsComplex(A) ;
}

inline static int
uIsRealScalar(const mxArray* A)
{
  return
    uIsReal (A) && mxGetNumberOfElements(A) == 1 ;
}

inline static mxArray *
newMxArrayFromDoubleVector (int n, double const* v)
{
  mxArray* array = mxCreateDoubleMatrix(n, 1, mxREAL) ;
  memcpy(mxGetPr(array), v, sizeof(double) * n) ;
  return (array) ;
}

inline static mxArray *
newMxArrayFromSvector (int n, SVECTOR const* sv)
{
  WORD* wi ;
  double *pr ;
  mwSize nz = 0 ;
  mwIndex *ir, *jc ;
  mxArray* sv_array ;

  /* count words */
  for (wi = sv->words ; wi->wnum >= 1 ; ++ wi) nz ++ ;

  /* allocate sparse array */
  sv_array = mxCreateSparse(n, 1, nz, mxREAL) ;
  /*  mxSetPr(mxMalloc(sizeof(double) * nz)) ;
   mxSetIr(mxMalloc(sizeof(mwIndex) * nz)) ;
   mxSetJc(mxMalloc(sizeof(mwIndex) * 2)) ;*/
  ir = mxGetIr (sv_array) ;
  jc = mxGetJc (sv_array) ;
  pr = mxGetPr (sv_array) ;

  /* copy fields */
  for (wi = sv->words ; wi->wnum >= 1 ; ++ wi) {
    *pr ++  = wi -> weight ;
    *ir ++  = wi -> wnum ;
    if (wi -> wnum > n) {
      char str [512] ;
      #ifndef WIN
      snprintf(str, sizeof(str),
               "Component index %d larger than sparse vector dimension %d",
               wi -> wnum, n) ;
      #else
      sprintf(str, sizeof(str),
               "Component index %d larger than sparse vector dimension %d",
               wi -> wnum, n) ;
      #endif
      mexErrMsgTxt(str) ;
    }
  }
  jc [0] = 0 ;
  jc [1] = nz ;

  return (sv_array) ;
}


typedef struct pattern {
  /*
    Type definition for input pattern x
  */
  mxArray * mex;
} PATTERN;

typedef struct label {
  /*
    Type definition for output label y
  */
  mxArray * mex;
} LABEL;

typedef struct latent_var {
  /*
    Type definition for latent variable h
  */
  mxArray * mex;
} LATENT_VAR;

typedef struct example {
  PATTERN x;
  LABEL y;
  LATENT_VAR h;
} EXAMPLE;

typedef struct sample {
  int n;
  EXAMPLE *examples;
} SAMPLE;


typedef struct structmodel {
  double *w;          /* pointer to the learned weights */
  MODEL  *svm_model;  /* the learned SVM model */
  long   sizePsi;     /* maximum number of weights in w */
  /* other information that is needed for the stuctural model can be
     added here, e.g. the grammar rules for NLP parsing */
  long n;
} STRUCTMODEL;


typedef struct struct_learn_parm {
  double epsilon;              /* precision for which to solve
				  quadratic program */
  long newconstretrain;        /* number of new constraints to
				  accumulate before recomputing the QP
				  solution */
  double C;                    /* trade-off between margin and loss */
  char   custom_argv[20][1000]; /* string set with the -u command line option */
  int    custom_argc;          /* number of -u command line options */
  int    slack_norm;           /* norm to use in objective function
                                  for slack variables; 1 -> L1-norm, 
				  2 -> L2-norm */
  int    loss_type;            /* selected loss function from -r
				  command line option. Select between
				  slack rescaling (1) and margin
				  rescaling (2) */
  int    loss_function;        /* select between different loss
				  functions via -l command line
				  option */
  /* add your own variables */
  mxArray  const * mex;
} STRUCT_LEARN_PARM;





inline static mxArray *
newMxArrayEncapsulatingDoubleVector (int n, double * v)
{
#if 1
  mxArray * v_array = mxCreateDoubleMatrix (0, 0, mxREAL) ;
  mxSetPr (v_array, v) ;
  mxSetM (v_array, n) ;
  mxSetN (v_array, 1) ;
  return v_array ;
#else
  return newMxArrayFromDoubleVector (n, v) ;
#endif
}

inline static mxArray *
newMxArrayEncapsulatingSmodel (STRUCTMODEL * smodel)
{
  mwSize dims [] = {1, 1} ;
  char const * fieldNames [] = {
    "w", "alpha", "svPatterns", "svLabels"
  } ;
  mxArray * smodel_array = mxCreateStructArray (2, dims, 4, fieldNames) ;

  /* we cannot just encapsulate the arrays because we need to shift by
   * one */

  mxSetField (smodel_array, 0, "w",
              newMxArrayFromDoubleVector
              (smodel->sizePsi, smodel->w + 1) ) ;

  return smodel_array ;
}

inline static void
destroyMxArrayEncapsulatingDoubleVector (mxArray * array)
{
  if (array) {
    mxSetN (array, 0) ;
    mxSetM (array, 0) ;
    mxSetPr (array, NULL) ;
    mxDestroyArray (array) ;
  }
}

inline static void
destroyMxArrayEncapsulatingSmodel (mxArray * array)
{
  if (array) {
    /* w and alpha are freed by mxDestroyArray, but we do not want this
     * to happen to the encapsulated patterns and labels yet (or are these shared?) */
    int i, n ;
    mxArray * svPatterns_array = mxGetField (array, 0, "svPatterns") ;
    mxArray * svLabels_array   = mxGetField (array, 0, "svLabels") ;
    if (svPatterns_array) {
      n = mxGetNumberOfElements (svPatterns_array) ;
      for (i = 0 ; i < n ; ++ i) {
        mxSetCell (svPatterns_array, i, NULL) ;
        mxSetCell (svLabels_array, i, NULL) ;
      }
    }
    mxDestroyArray (array) ;
  }
}
