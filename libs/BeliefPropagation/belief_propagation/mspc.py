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
import sys,os
from igraph import *
import numpy as np
assert(len(sys.argv)>2 and "./mspc.py fname Q [output ovl]")
fname=sys.argv[1]
Q=int(sys.argv[2])
g=Graph.Read_GML(fname)
try:
	y=g.vs['value']
	y=[int(i) for i in y]
except:
	pass
z=g.community_leading_eigenvector(Q).membership
ovl=0
try:
	xx=np.array([1 if z[i]==y[i] else 0 for i in range(len(y))])
	ovl=1.0*xx.sum()/len(xx)
except:
	pass
if len(sys.argv)>=4:
	print ovl
for i in z: sys.stdout.write(str(i)+" ")
sys.stdout.write("\n")


