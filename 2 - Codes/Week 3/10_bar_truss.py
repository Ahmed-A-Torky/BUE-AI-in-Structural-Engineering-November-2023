# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 01:11:55 2020

@author: Ahmed_A_Torky

The material used in this structure has a modulus of elasticity of and a mass
density of γ = 2.768T/m3 . The design constraints are: maximum allowable stress in any member of
the truss σallowable = ±0.172MPa ; maximum deflection of any node (in both vertical and horizontal
directions) δallowable = 50.8mm . The upper and lower limits of cross-section areas of each truss
member 0.645mm2 ≤ Ai ≤ 225.8mm2 . The sample points by which the sample grid is generated are
1.0 and 30.0.
"""


import openseespy.opensees as op
import numpy as np
# =============================================================================
# Units
# =============================================================================
mm = 1.
N = 1.
Pa = 1.
inch = 25.4*mm
m = 1000.*mm
kN = 1000.*N
MPa = 1. # (10.**6)*Pa
GPa = (10.**3)*MPa
ton = 9.80665*kN
# =============================================================================
# Input Variables
# =============================================================================
x1 = 0.
y1 = 0.
x2 = 0.
y2 = 9.144*m
x3 = 9.144*m
y3 = 0.
x4 = 9.144*m
y4 = 9.144*m
x5 = 9.144*m * 2
y5 = 0.
x6 = 9.144*m * 2
y6 = 9.144*m

# Section Area (0.645mm**2 ≤ Ai ≤ 225.8mm**2)
A1 = 35.0*inch**2 # 225.8 # mm**2
A2 = 0.10*inch**2 # 0.645 # mm**2
# Modulus of elasticity
E = 68.948*GPa # 68947.5908 MPa
# Loads
P3 = -45.359*ton # 444.82 kN
P5 = -45.359*ton # 444.82 kN
# Weight
gamma = 2.768*ton/(m**3)
# Loop on random Areas
import random
A = []
for x in range(20):
  A.append(random.uniform(A2,A1))
print (A)
# Lengths
L = []
for i in range(6):
    L.append(9.144*m)
for i in range(4):
    L.append(np.sqrt((9.144*m)**2 + (9.144*m)**2))
# Weight
TotalWegiht = 0.0
for ii in range(10):
    TotalWegiht += gamma*L[ii]*A[ii]
# =============================================================================
# OpenSees Analysis
# =============================================================================
# remove existing model
op.wipe()
# set modelbuilder
op.model('basic', '-ndm', 2, '-ndf', 3)
# define materials
op.uniaxialMaterial("Elastic", 1, E)
# create nodes
op.node(1, x1, y1)
op.node(2, x2, y2)
op.node(3, x3, y3)
op.node(4, x4, y4)
op.node(5, x5, y5)
op.node(6, x6, y6)
# set boundary condition
op.fix(1, 1, 1, 1)
op.fix(2, 1, 1, 1)
op.fix(3, 0, 0, 1)
op.fix(4, 0, 0, 1)
op.fix(5, 0, 0, 1)
op.fix(6, 0, 0, 1)
# define elements
# op.element('Truss', eleTag, *eleNodes, A, matTag[, '-rho', rho][, '-cMass', cFlag][, '-doRayleigh', rFlag])
op.element("Truss", 1, 1, 3, A[0], 1)
op.element("Truss", 2, 3, 5, A[1], 1)
op.element("Truss", 3, 5, 6, A[2], 1)
op.element("Truss", 4, 6, 4, A[3], 1)
op.element("Truss", 5, 4, 2, A[4], 1)
op.element("Truss", 6, 3, 4, A[5], 1)
op.element("Truss", 7, 1, 4, A[6], 1)
op.element("Truss", 8, 2, 3, A[7], 1)
op.element("Truss", 9, 3, 6, A[8], 1)
op.element("Truss", 10, 5, 4, A[9], 1)
# create TimeSeries
op.timeSeries("Linear", 1)
# create a plain load pattern
op.pattern("Plain", 1, 1)
# Create the nodal load - command: load nodeID xForce yForce
# op.load(4, Px, Py, 0.)
op.load(3, 0., P3, 0.)
op.load(5, 0., P5, 0.)
# Record Results
nodes = [1,2,3,4,5,6]
bnodes = [1,2]
elements = [1,2,3,4,5,6,7,8,9,10]
op.recorder('Node', '-file', "Node3Disp.out", '-time', '-node', 3, '-dof', 1,2,3,'disp')
op.recorder('Node', '-file', "Node5Disp.out", '-time', '-node', 5, '-dof', 1,2,3,'disp')
op.recorder('Node', '-file', "Reaction.out", '-time', '-node', *bnodes, '-dof', 1,2,3,'reaction')
op.recorder('Element', '-file', "Elements.out",'-time','-ele', *elements, 'forces')
# create SOE
op.system("BandSPD")
# create DOF number
op.numberer("RCM")
# create constraint handler
op.constraints("Plain")
# create integrator
op.integrator("LoadControl", 1.0)
# create algorithm
op.algorithm("Newton")
# create analysis object
op.analysis("Static")
# perform the analysis
op.initialize() 
ok = op.analyze(1)
# Check the vl & hl results of displacement
ux = []
uy = []
for i in range(6):
    ux.append(op.nodeDisp(i+1,1))
    uy.append(op.nodeDisp(i+1,2))
u3 = op.nodeDisp(5,2)
u5 = op.nodeDisp(3,2)
print("nodeDisp 5 =",u5,"mm")
print("nodeDisp 3 =",u3,"mm")
print("x DISP =",ux,"mm")
print("y DISP =",uy,"mm")
print("Weight =",TotalWegiht,"N")
# Must reset
op.wipe()
Weights = []
TotalWegiht = 0.0
for ii in range(10):
    TotalWegiht += gamma*L[ii]*A[ii]
if any(x >= 50.8 for x in np.abs(ux)):
    print("ux exceeded 50.8mm")
    Weights.append(1000000)
elif any(y >= 50.8 for y in np.abs(uy)):
    print("uy exceeded 50.8mm")
    Weights.append(1000000)
else:
    Weights.append(TotalWegiht)
