# Makefile for Latent Structural SVM for matlab

CC=gcc
#CFLAGS= -g -Wall
CFLAGS= -O3 -fomit-frame-pointer -ffast-math -fPIC -I/usr/local/MATLAB/R2016a/extern/include/
#CFLAGS = -O3 -pg
LD=gcc
#LDFLAGS= -g
LDFLAGS= -O3
#LDFLAGS = -O3 -pg
LIBS= -lm

Darwin_PPC_ARCH := mac
Darwin_Power_Macintosh_ARCH := mac
Darwin_i386_ARCH := maci64
Darwin_x86_64_ARCH := maci64
Linux_i386_ARCH := glnx86
Linux_i686_ARCH := glnx86
Linux_unknown_ARC := glnx86
Linux_x86_64_ARCH := glnxa64


UNAME := $(shell uname -sm)
ARCH ?= $($(shell echo "$(UNAME)" | tr \  _)_ARCH)
MEX ?= mex
# Linux-64
ifeq ($(ARCH),glnxa64)
LDFLAGS +=
MEXEXT = mexa64
endif


MEXFLAGS += -largeArrayDims -$(ARCH) CFLAGS='$$CFLAGS $(CFLAGS) -Wall -std=c99' LDFLAGS='$$LDFLAGS $(LDFLAGS)' 


all: svm_latent_struct_learn_mex.mexa64

clean: 
	rm -f *.o
	rm -f svm_latent_struct_learn_mex.mexa64

svm_light_hideo_noexe: 
	cd svm_light; make svm_learn_hideo_noexe

svm_struct_latent_api.o: svm_struct_latent_api.c svm_struct_latent_api_types.h
	$(CC) -c $(CFLAGS) svm_struct_latent_api.c -o svm_struct_latent_api.o

svm_latent_struct_learn_mex.mexa64: svm_latent_struct_learn_mex.c svm_light/svm_hideo.o svm_light/svm_learn.o svm_light/svm_common.o svm_struct_latent_api.o -lm
	$(MEX) $(MEXFLAGS) $^ -output "$@"	