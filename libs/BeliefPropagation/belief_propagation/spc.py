#!/usr/bin/python
# sbm version 1.1, release date 30/05/2012
# Copyright 2012 Aurelien Decelle, Florent Krzakala, Lenka Zdeborova and Pan Zhang
# sbm is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or 
# (at your option) any later version.
#
# sbm is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import sys
import numpy as np
import scipy.sparse.linalg as linalg
from scipy import sparse
from scipy.cluster.vq import kmeans2
from scipy.linalg import eig

mode=3
N=-1
assert(len(sys.argv)>=4 and "too few parameters. Usage: spc.py N filename q [mode]")
N=int(sys.argv[1])
fname=sys.argv[2]
K=int(sys.argv[3])
mode=int(sys.argv[4])
assert(N>0)
assert(mode in [0,1,2,3,5,6,7] and "wrong mode, mode should be 0, 1, 2 or 3")
try:
	fin=open(fname,'r')
except:
	print "can not open "+fname
	sys.exit(0)
lines=fin.readlines()
l1=[int(line.split()[1]) for line in lines]
l0=[int(line.split()[0]) for line in lines]
x0=np.array(l0+l1)
x1=np.array(l1+l0)
A=sparse.csr_matrix( (np.ones(len(lines)*2),( x0,x1 )),shape=(N,N)) # adjacent matrix in sparse representation
m=np.array(A.mean(1))[:,0]*N # degrees of A in vector
if mode == 0 : # K smallest eigenvecotors of Laplacian
	D=sparse.csc_matrix((m,(np.arange(N),np.arange(N))),shape=(N,N)) # degree of A in diagonal matrix
	L=D-A # graph Laplacian
	#x,y=linalg.eigs(L,K,sigma=0,which='LM') # K smarlest eignvectors
	x,y=linalg.eigs(L,K,which='LR') # K smarlest eignvectors
if mode == 1 : # K smallest eigenvectors of Lrw
	D=sparse.csr_matrix((m,(np.arange(N),np.arange(N))),shape=(N,N))
	L=D-A
	m1=m.copy() # 1.0/degree of A in vector
	m1[m>0]=1.0/m1[m>0]
	D1=sparse.csc_matrix((m1,(np.arange(N),np.arange(N))),shape=(N,N)) # 1.0/degree of A in diagonal matrix
	Lrw=D1*L
	#x,y=linalg.eigs(Lrw,K,sigma=0,which='LM')
	x,y=linalg.eigs(Lrw,K,which='SR')
if mode == 2 : # K smallest eigenvectors of Lsym
	D=sparse.csc_matrix((m,(np.arange(N),np.arange(N))),shape=(N,N)) # degree of A in diagonal matrix
	L=D-A
	m2=m.copy() # 1.0/sqrt(degree) of A in vector
	m2[m>0]=1.0/np.sqrt(m[m>0])
	D2=sparse.csc_matrix((m2,(np.arange(N),np.arange(N))),shape=(N,N)) # 1.0/sqrt(degree) of A in diagonal matrix
	Lsym=D2*L*D2
	#x,y=linalg.eigs(Lsym,K,sigma=0,which='LM')
	x,y=linalg.eigs(Lsym,K,which='SR')
if mode == 3 :# K largest eigen values of P
	m1=m.copy()
	m1[m>0]=1.0/m1[m>0]
	D1=sparse.csr_matrix((m1,(np.arange(N),np.arange(N))),shape=(N,N)) # 1.0/degree of A in diagonal matrix
	P=D1*A
	x,y=linalg.eigs(P,K,which='LM')
if mode == 5 :# The second largest eigen vector of A
#	m1=m.copy()
#	m1[m>0]=1.0/m1[m>0]
#	D1=sparse.csr_matrix((m1,(np.arange(N),np.arange(N))),shape=(N,N)) # 1.0/degree of A in diagonal matrix
#	P=D1*A
	P=A
	x,y=linalg.eigs(P,2,which='LR')
#	print x[1]
	z=[i[1] for i in y]
	z=np.real(z)
#	print z
	zz=[1 if i>0 else 0 for i in z]
	for i in zz: sys.stdout.write(str(i)+" ")
	sys.stdout.write("\n")
	sys.exit(0)
#	for i in np.real(y): print i
if mode == 6 :# The second smallest eigen vector of L
	m1=m.copy()
	m1[m>0]=1.0/m1[m>0]
	D1=sparse.csr_matrix((m1,(np.arange(N),np.arange(N))),shape=(N,N)) # 1.0/degree of A in diagonal matrix
	P=D1*A
	x,y=linalg.eigs(P,2,which='LR')
#	print x[1]
	z=[i[1] for i in y]
	z=np.real(z)
#	print z
	zz=[1 if i>0 else 0 for i in z]
	for i in zz: sys.stdout.write(str(i)+" ")
	sys.stdout.write("\n")
	sys.exit(0)

	
centroids,groups=kmeans2(np.real(y),K)
for i in groups: sys.stdout.write(str(i)+" ")
sys.stdout.write("\n")

