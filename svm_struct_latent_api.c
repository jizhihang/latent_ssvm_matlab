/************************************************************************/
/*                                                                      */
/*   svm_struct_latent_api.c                                            */
/*                                                                      */
/*   API function definitions for Latent SVM^struct                     */
/*                                                                      */
/*   Author: Chun-Nam Yu                                                */
/*   Date: 17.Dec.08                                                    */
/*                                                                      */
/*   This software is available for non-commercial use only. It must    */
/*   not be modified and distributed without prior permission of the    */
/*   author. The author is not responsible for implications from the    */
/*   use of this software.                                              */
/*                                                                      */
/************************************************************************/

#include <stdio.h>
#include <assert.h>
#include "svm_struct_latent_api_types.h"

#define MAX_INPUT_LINE_LENGTH 10000

SAMPLE read_struct_examples(mxArray const * sparm_array, STRUCT_LEARN_PARM *sparm) {
/*
  Read input examples {(x_1,y_1),...,(x_n,y_n)} from file.
  The type of pattern x and label y has to follow the definition in 
  svm_struct_latent_api_types.h. Latent variables h can be either
  initialized in this function or by calling init_latent_variables(). 
*/
  int numExamples, ei;
  mxArray const * patterns_array;
  mxArray const * labels_array;
  mxArray const * latent_array;
  patterns_array = mxGetField(sparm_array, 0, "patterns");
  if (!patterns_array || !mxIsCell(patterns_array))
    mexErrMsgTxt("SPARM.PATTERNS must be a cell array");

  numExamples = mxGetNumberOfElements(patterns_array);

  labels_array = mxGetField(sparm_array, 0, "labels") ;
  if (!labels_array || !mxIsCell(labels_array) || !mxGetNumberOfElements(labels_array) == numExamples)
    mexErrMsgTxt("SPARM.LABELS must be a cell array "
               "with the same number of elements of "
               "SPARM.PATTERNS");

  latent_array = mxGetField(sparm_array, 0, "labels_latent") ;
  if (!latent_array || !mxIsCell(latent_array) || !mxGetNumberOfElements(latent_array) == numExamples)
    mexErrMsgTxt("SPARM.LABELS_LATENT must be a cell array "
               "with the same number of elements of "
               "SPARM.PATTERNS");

  SAMPLE sample;
  sample.n = numExamples;
  sample.examples = (EXAMPLE *) my_malloc (sizeof(EXAMPLE) * numExamples) ;
  for (ei = 0 ; ei < numExamples ; ++ ei)
  {
    sample.examples[ei].x.mex = mxGetCell(patterns_array, ei);
    sample.examples[ei].y.mex = mxGetCell(labels_array,   ei);
    sample.examples[ei].h.mex = mxGetCell(latent_array,  ei);
  }

  mexPrintf("There are %d training examples\n", numExamples) ;
  return sample;
}

void init_struct_model(SAMPLE sample, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, LEARN_PARM *lparm, KERNEL_PARM *kparm) {
/*
  Initialize parameters in STRUCTMODEL sm. Set the diminension 
  of the feature space sm->sizePsi. Can also initialize your own
  variables in sm here. 
*/

  // assume linear kernel
  mxArray const * sizePsi_array = mxGetField(sparm->mex, 0, "dimension") ;
  if (! sizePsi_array) {
      mexErrMsgTxt("Field PARM.DIMENSION not found") ;
  }
  if (! uIsRealScalar(sizePsi_array)) {
    mexErrMsgTxt("PARM.DIMENSION must be a scalar") ;
  }

  sm->sizePsi = *mxGetPr(sizePsi_array) ;
  if (sm->sizePsi < 1) {
    mexErrMsgTxt("PARM.DIMENSION must be not smaller than 1") ;
  }
}

void init_latent_variables(SAMPLE *sample, LEARN_PARM *lparm, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Initialize latent variables in the first iteration of training.
  Latent variables are stored at sample.examples[i].h, for 1<=i<=sample.n.
*/
}

SVECTOR *psi(PATTERN x, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Creates the feature vector \Psi(x,y,h) and return a pointer to 
  sparse vector SVECTOR in SVM^light format. The dimension of the 
  feature vector returned has to agree with the dimension in sm->sizePsi. 
*/
//  printf("begin psi()\n");
  SVECTOR *sv=NULL;
  mxArray* out ;
  mxArray* fn_array ;
  mxArray* args [5] ;
  WORD* words = NULL ;
  int status ;

  fn_array = mxGetField(sparm->mex, 0, "featureFn") ;
  if (!fn_array)
    mexErrMsgTxt("Field PARM.FEATUREFN not found") ;

  if (!mxGetClassID(fn_array) == mxFUNCTION_CLASS)
    mexErrMsgTxt("PARM.FEATUREFN must be a valid function handle") ;

  args[0] = fn_array;
  args[1] = (mxArray*) sparm->mex; /* model (discard conts) */
  args[2] = x.mex;                 /* pattern */
  args[3] = y.mex;                 /* label */
  args[4] = h.mex;                 /* latent */
  status = mexCallMATLAB(1, &out, 5, args, "feval") ;
  if (status)
    mexErrMsgTxt("Error while executing PARM.FEATUREFN") ;

  if (mxGetClassID(out) == mxUNKNOWN_CLASS)
    mexErrMsgTxt("PARM.FEATUREFN must reutrn a result") ;


  if (! mxIsSparse(out) || ! mxGetClassID(out) == mxDOUBLE_CLASS ||
      ! mxGetN(out) == 1 || ! mxGetM(out) == sm->sizePsi)
    mexErrMsgTxt("PARM.FEATUREFN must return a sparse column vector "
                 "of the prescribed size") ;

  double * data = mxGetPr(out) ;
  int i ;
  mwIndex * colOffsets = mxGetJc(out) ;
  mwIndex * rowIndexes = mxGetIr(out) ;
  int numNZ = colOffsets[1] - colOffsets[0] ;
  words = (WORD*) my_malloc (sizeof(WORD) * (numNZ + 1)) ;

  for (i = 0 ; i < numNZ ; ++ i) {
    words[i].wnum = rowIndexes[i] + 1 ;
    words[i].weight = data[i] ;
  }
  words[numNZ].wnum = 0 ;
  words[numNZ].weight = 0 ;

  sv = create_svector(words, "", 1);
  free(words);
  mxDestroyArray(out) ;
//  printf("end psi()\n");

  return sv;
}



void classify_struct_example(PATTERN x, LABEL *y, LATENT_VAR *h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Makes prediction with input pattern x with weight vector in sm->w,
  i.e., computing argmax_{(y,h)} <w,psi(x,y,h)>. 
  Output pair (y,h) are stored at location pointed to by 
  pointers *y and *h. 
*/
}


void find_most_violated_constraint_marginrescaling(PATTERN x, LABEL y, LABEL *ybar, LATENT_VAR *hbar, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Finds the most violated constraint (loss-augmented inference), i.e.,
  computing argmax_{(ybar,hbar)} [<w,psi(x,ybar,hbar)> + loss(y,ybar,hbar)].
  The output (ybar,hbar) are stored at location pointed by 
  pointers *ybar and *hbar. 
*/
//  printf("begin find_most_violated_constraint_marginrescaling()\n");
  mxArray* fn_array ;
  mxArray* model_array ;
  mxArray* args [5] ;
  mxArray* out[2];
  int status ;

  fn_array = mxGetField(sparm->mex, 0, "constraintFn") ;
  if (! fn_array)
   mexErrMsgTxt("Field PARM.CONSTRAINTFN not found");

  if (! mxGetClassID(fn_array) == mxFUNCTION_CLASS)
   mexErrMsgTxt("PARM.CONSTRAINTFN is not a valid function handle") ;

  /* encapsulate sm->w into a Matlab array */
  model_array = newMxArrayEncapsulatingSmodel (sm) ;

  args[0] = fn_array ;
  args[1] = (mxArray*) sparm->mex ;
  args[2] = model_array;
  args[3] = x.mex;
  args[4] = y.mex;

  status = mexCallMATLAB(2, out, 5, args, "feval") ;
  destroyMxArrayEncapsulatingSmodel (model_array) ;

  if (status)
  {
    mxArray * error_array ;
    mexCallMATLAB(1, &error_array, 0, NULL, "lasterror") ;
    mexCallMATLAB(0, NULL, 1, &error_array, "error") ;
  }

  if (mxGetClassID(*out) == mxUNKNOWN_CLASS)
      mexErrMsgTxt("PARM.CONSTRAINTFN did not return a result") ;


  ybar->mex = out[0];
  hbar->mex = out[1];
//printf("end find_most_violated_constraint_marginrescaling()\n");


}

LATENT_VAR infer_latent_variables(PATTERN x, LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Complete the latent variable h for labeled examples, i.e.,
  computing argmax_{h} <w,psi(x,y,h)>. 
*/
//    printf("begin infer_latent_variables()\n");
  LATENT_VAR h;
  mxArray* fn_array ;
  mxArray* model_array ;
  mxArray* args [5] ;
  mxArray* out;
  int status ;

  fn_array = mxGetField(sparm->mex, 0, "inferLatentFn") ;
  if (! fn_array)
     mexErrMsgTxt("Field PARM.INFERLATENTFNis not found");

  if (! mxGetClassID(fn_array) == mxFUNCTION_CLASS)
    mexErrMsgTxt("PARM.INFERLATENTFNis not a valid function handle") ;


  /* encapsulate sm->w into a Matlab array */
  model_array = newMxArrayEncapsulatingSmodel (sm) ;

  args[0] = fn_array ;
  args[1] = (mxArray*) sparm->mex ; /* model (discard conts) */
  args[2] = model_array;
  args[3] = x.mex;
  args[4] = y.mex;

  status = mexCallMATLAB(1, &out, 5, args, "feval") ;
  destroyMxArrayEncapsulatingSmodel (model_array) ;

  if (status)
  {
    mxArray * error_array ;
    mexCallMATLAB(1, &error_array, 0, NULL, "lasterror") ;
    mexCallMATLAB(0, NULL, 1, &error_array, "error") ;
  }

  if (mxGetClassID(out) == mxUNKNOWN_CLASS)
     mexErrMsgTxt("PARM.CONSTRAINTFN did not reutrn a result") ;
//    printf("end infer_latent_variables()\n");
  h.mex = out;
  return h;
}


double loss(LABEL y, LABEL ybar, LATENT_VAR hbar, STRUCT_LEARN_PARM *sparm) {
/*
  Computes the loss of prediction (ybar,hbar) against the
  correct label y. 
*/
//        printf("begin loss()\n");
    double loss_value ;
    mxArray* fn_array ;
    mxArray* out ;
    mxArray* args [5];
    int status ;

    fn_array = mxGetField(sparm->mex, 0, "lossFn") ;
    if (! fn_array)
      mexErrMsgTxt("Field PARM.LOSSFN not found") ;

    if (! mxGetClassID(fn_array) == mxFUNCTION_CLASS)
      mexErrMsgTxt("PARM.LOSSFN must be a valid function handle") ;


    args[0] = fn_array ;
    args[1] = (mxArray*) sparm->mex ; /* model (discard conts) */
    args[2] = y.mex;
    args[3] = ybar.mex;
    args[4] = hbar.mex;

    status = mexCallMATLAB (1, &out, 5, args, "feval") ;

    if (status)
      mexErrMsgTxt("Error while executing PARM.LOSSFN") ;

    if (!uIsRealScalar(out))
      mexErrMsgTxt("PARM.LOSSFN must reutrn a scalar") ;

    loss_value = *mxGetPr(out);
    mxDestroyArray(out) ;
//    printf("end loss()\n");
    return (loss_value) ;
}

void write_struct_model(char *file, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Writes the learned weight vector sm->w to file after training. 
*/
    FILE *modelfl;
    int i;

    modelfl = fopen(file,"w");
    if (modelfl==NULL) {
      printf("Cannot open model file %s for output!", file);
      exit(1);
    }
    for (i=1;i<sm->sizePsi+1;i++) {
      fprintf(modelfl, "%d:%.16g\n", i, sm->w[i]);
    }
    fclose(modelfl);    
}

STRUCTMODEL read_struct_model(char *file, STRUCT_LEARN_PARM *sparm) {
/*
  Reads in the learned model parameters from file into STRUCTMODEL sm.
  The input file format has to agree with the format in write_struct_model().
*/
  STRUCTMODEL sm;

  /* your code here */

  return(sm);
}

void free_struct_model(STRUCTMODEL sm, STRUCT_LEARN_PARM *sparm) {
/*
  Free any memory malloc'ed in STRUCTMODEL sm after training. 
*/

  /* your code here */
  
  free(sm.w);
}

void free_pattern(PATTERN x) {
/*
  Free any memory malloc'ed when creating pattern x. 
*/

  /* your code here */

}

void free_label(LABEL y) {
/*
  Free any memory malloc'ed when creating label y. 
*/

  /* your code here */

} 

void free_latent_var(LATENT_VAR h) {
/*
  Free any memory malloc'ed when creating latent variable h. 
*/

  /* your code here */

}

void free_struct_sample(SAMPLE s) {
/*
  Free the whole training sample. 
*/
  int i;
  for (i=0;i<s.n;i++) {
    free_pattern(s.examples[i].x);
    free_label(s.examples[i].y);
    free_latent_var(s.examples[i].h);
  }
  free(s.examples);

}

void parse_struct_parameters(STRUCT_LEARN_PARM *sparm) {
/*
  Parse parameters for structured output learning passed 
  via the command line. 
*/
  int i;
  
  /* set default */
  
  for (i=0;(i<sparm->custom_argc)&&((sparm->custom_argv[i])[0]=='-');i++) {
    switch ((sparm->custom_argv[i])[2]) {
      /* your code here */
      default: printf("\nUnrecognized option %s!\n\n", sparm->custom_argv[i]); exit(0);
    }
  }
}

