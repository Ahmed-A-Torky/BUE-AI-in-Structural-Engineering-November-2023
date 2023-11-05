# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 18:56:12 2020
Must install:
    pip install geneticalgorithm
    pip install openseespy
@author: Ahmed_A_Torky
"""
import openseespy.opensees as op
import numpy as np
import matplotlib.pyplot as plt
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
lb = 4.4482*N
# =============================================================================
# Input Variables
# =============================================================================
# Node Coordinates
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
E = 68.94757*GPa # 68947.5908 MPa
# Loads
P3 = -45.359*ton 
P5 = -45.359*ton 
# Weight
gamma = 2.76799*ton/(m**3)
# Lengths
L = []
for i in range(6):
    L.append(9.144*m)
for i in range(4):
    L.append(np.sqrt((9.144*m)**2 + (9.144*m)**2))
# =============================================================================
# Subroutine for 10-bar truss calculations
# =============================================================================
# Function of 10-bar truss
def tenbar_truss(x):
    x = np.array(x)
    # print(x)
    # u = []
    Weights = []
    A=x.tolist()
    # print(A)
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
    # No need to Record Results (writing takes time)
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
    # Must reset
    op.wipe()
    TotalWegiht = 0.0
    for ii in range(10):
        TotalWegiht += gamma*L[ii]*A[ii]
    if any(x >= 50.8 for x in np.abs(ux)):
        # print("ux exceeded 50.8mm")
        Weights.append(100000)
    elif any(y >= 50.8 for y in np.abs(uy)):
        # print("uy exceeded 50.8mm")
        Weights.append(100000)
    else:
        Weights.append(TotalWegiht)
    # u = np.array(u)
    Weights = np.array(Weights)
    # print(Weights/lb)
    return Weights/lb
# =============================================================================
#                                   GA
# =============================================================================
# Genetic Algorithm Part
from geneticalgorithm import geneticalgorithm as ga
# Set the section bounds
x_max = A1 * np.ones(10)
x_min = A2 * np.ones(10)
varbound=np.array([[A2,A1]]*10)
# instatiate the optimizer
algorithm_param = {'max_num_iteration': 10000,\
                   'population_size':10000,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.2,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':1000}
# Define the model
model=ga(function=tenbar_truss,\
            dimension=10,\
            variable_type='real',\
            variable_boundaries=varbound,\
            algorithm_parameters=algorithm_param)
# Run the GA model
model.run()
# Try to get Weight less than 5100 lb (competitive)
print("Model parameters are:",algorithm_param)
print("Final Weight is:","{:.2f}".format(round(model.best_function, 2)),"lb")
print("Best cross-sections are:",model.best_variable/inch**2,"inches")