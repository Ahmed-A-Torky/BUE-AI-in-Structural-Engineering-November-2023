# -*- coding: utf-8 -*-
"""
Created on Mon Aug 01 12:59:59 2023
make sure you install:
    pip install geneticalgorithm
    pip install geneticalgorithm2
    pip install comtypes

@author: Ahmed A. Torky
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import comtypes.client
import pythoncom
import datetime
aa = datetime.datetime.now()


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

filename1 = 'Runs_GA_Sym/Origin.$2k' # INPUT
filename2 = 'Runs_GA_Sym/Particle.$2k' # OUTPUT
filename3 = 'Runs_GA_Sym/Best.$2k' # OUTPUT

# Find the "target keyword" and return the line number of the
# line below the line where the target keyword is located
# =============================================================================
# Subroutine for searching $2k
# =============================================================================
def SearchKeyword(keyword, lines):
    for index, line in enumerate(lines):
        if keyword in line:
            # The target keyword is in the index line of the file 
            # (note that counting starts from 0 in python, that is, the index 
            # of the first line in the file is 0)
            # The line number of the line below the line where the target 
            # keyword is located is index + 1
            startline = index + 1
            break
    else:
        startline = -1
    return startline


# =============================================================================
# Subroutine for Run SAP2000
# =============================================================================
def RunSAP2000(filename2):
    pythoncom.CoInitialize()
    # open an existing $2k file
    FileName = filename2
    a = datetime.datetime.now()
    
    AttachToInstance = False
    SpecifyPath = False
    #full path to the model
    script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
    # print(script_directory)
    APIPath = script_directory+'\Runs_GA_Sym'
    if not os.path.exists(APIPath):
            try:
                os.makedirs(APIPath)
            except OSError:
                pass
    #TODO No need to save sdb for now
    # ModelPath = APIPath + os.sep + 'Trial1.sdb'
    
    #create API helper object
    helper = comtypes.client.CreateObject('SAP2000v1.Helper')
    helper = helper.QueryInterface(comtypes.gen.SAP2000v1.cHelper)
    
    if AttachToInstance:
        #attach to a running instance of SAP2000
        try:
            #get the active SapObject
                mySapObject = helper.GetObject("CSI.SAP2000.API.SapObject") 
        except (OSError, comtypes.COMError):
            print("No running instance of the program found or failed to attach.")
            sys.exit(-1)
    else:
        if SpecifyPath:
            try:
                #'create an instance of the SAPObject from the specified path
                mySapObject = helper.CreateObject(ProgramPath)
            except (OSError, comtypes.COMError):
                print("Cannot start a new instance of the program from " + ProgramPath)
                sys.exit(-1)
        else:
            try:
                #create an instance of the SAPObject from the latest installed SAP2000
                mySapObject = helper.CreateObjectProgID("CSI.SAP2000.API.SapObject")
            except (OSError, comtypes.COMError):
                print("Cannot start a new instance of the program.")
                sys.exit(-1)
        #start SAP2000 application
        mySapObject.ApplicationStart()
    
    #create SapModel object
    SapModel = mySapObject.SapModel
    # initialize model
    ret = SapModel.InitializeNewModel()
    # open an existing file
    ret = SapModel.File.OpenFile(FileName)
    # #save model
    # ret = SapModel.File.Save(ModelPath)
    #run model (this will create the analysis model)
    ret = SapModel.Analyze.RunAnalysis()
    # Get Sap2000 results
    ObjectElm = 0
    Element = 1
    GroupElm = 2
    SelectionElm = 3
    NumberResults = 0
    Obj = []
    Elm = []
    #TODO choose your loadcase
    # ACase = []
    ACase = 'Live'
    StepType = []
    StepNum = []
    U1 = []
    U2 = []
    U3 = []
    R1 = []
    R2 = []
    R3 = []
    # clear all case and combo output selections
    ret = SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput()
    # set case and combo output selections
    ret = SapModel.Results.Setup.SetCaseSelectedForOutput('Live')
    # get point displacements
    # e.g. : [NumberResults, Obj, Elm, ACase, StepType, StepNum, U1, U2, U3, R1, R2, R3, ret] = SapModel.Results.JointDispl('13', ObjectElm, NumberResults, Obj, Elm, ACase, StepType, StepNum, U1, U2, U3, R1, R2, R3)
    ALL_U3 = []
    for i in range(n_frames):
        [NumberResults, Obj, Elm, ACase, StepType, StepNum, U1, U2, U3, R1, R2, R3, ret] = SapModel.Results.JointDispl(str(i+1), ObjectElm, NumberResults, Obj, Elm, ACase, StepType, StepNum, U1, U2, U3, R1, R2, R3)
        ALL_U3.append(U3)
    
    #close Sap2000
    ret = mySapObject.ApplicationExit(False)
    SapModel = None
    mySapObject = None
    
    # print(ALL_U1)
    # print(ALL_U3)
    
    b = datetime.datetime.now()
    c = b - a
    
    print("This run lasted:",c,"hrs:mins:seconds")
    return ALL_U3

# =============================================================================
# Subroutine for truss calculations
# =============================================================================
# Function of truss
def truss(x):
    Weights = []
    A=x.tolist()
    A_actual = []
    print(A)
    # =============================================================================
    # MODIFY $2K according to suggestion x
    # =============================================================================
    # ------------------------------------------------------------------------------
    # CHANGING FRAME SECTION ASSIGNMENTS
    # The main body of the interface program, 
    # open the $2k file and call it f for short
    # ------------------------------------------------------------------------------
    # Open Pandas AISC sections, check the nearest area of section from x,
    df = pd.read_csv('Runs_GA/TUBO.csv')
    # temp = df[df['Area'] >= 0.00073].iloc[0]
    # Design_Section = temp['SectionName']
    print("Assumed sections in particle/chromosome are: ")

    # get its name, place it in F3
    with open(filename1) as f:
        lines = f.readlines()
        a = SearchKeyword('FRAME SECTION ASSIGNMENTS', lines)
        if a == -1:
            print('Frame Sections Definition NOT Found')
        else:
            frame_total = 0
            for ij,line in enumerate(lines[a:a+n_frames]):
                # print(ij)
                Area_Required = A[ij]
                if not line.isspace():
                    result = line.split()
                    F1 = result[0].split('=')[1]
                    F2 = result[1].split('=')[1]
                    F3 = result[2].split('=')[1]
                    F4 = result[3].split('=')[1]
                    
                    # Modify Sections from BEam 250x700 to  BEam260x700
                    """ 
                    IMPORTANT NOTE!
                    DO NOT name your sections with spaces in their names
                    e.g. of NOT TO DO "BEam 260x700"
                    e.g. of accepted name "BEam260x700"
                    """
                    # Find closest section to Area_Required and place it in F3
                    # ...
                    temp = df[df['Area'] >= Area_Required].iloc[0]
                    Design_Section = temp['SectionName']
                    A_actual.append(temp['Area']*m**2)
                    
                    F3 = Design_Section
                    
                    print(F3)
                    
                    """e.g. : Frame=1   AutoSelect=N.A.   AnalSect="BEam250x700"   MatProp=Default"""
                    lines[a+ij] = '   Frame='+str(F1)+'   AutoSelect='+str(F2)+\
                        '   AnalSect='+str(F3)+'   MatProp='+str(F4)+'\n'
                    frame_total += 1
                else:
                    print('Frame Sections Assignments have Been Updated')
                    break
            print("All frames assigned, they are:", frame_total)
                

    # Open the output file for writing
    with open(filename2, "w") as output_file:
        # Write the new contents to the output file
        output_file.writelines(lines)
        print('Frame Sections Assignments have Been Written')

    # =============================================================================
    # SAP2000 Analysis
    # =============================================================================
    # Run Sap2000 AND collect the deflections
    ALL_U3 = RunSAP2000(filename2)
    uz_13 = ALL_U3[12]
    
    # =============================================================================
    # Check Weights (Fitness Function)
    # =============================================================================
    TotalWegiht = 0.0
    #TODO change lengths of each member
    L = 4 * m # change
    # A = 10 # I changed it to A_actual
    gamma = 78.49*kN/m**3 # change
    print('A_actual:',A_actual)
    print('Length:',L)
    print('gamma:',gamma)
    
    # Check Uz against code limits
    print('uz 13 =',np.abs(uz_13)[0]*m )
    print('limit =', 30*m/360)
    #TODO Change all weights according to total Truss Weight
    for ii in range(n_frames):
        TotalWegiht += gamma*L*(A_actual[ii]) # *L[ii]*A[ii]
    if np.abs(uz_13)[0]*m  >= 30*m/360:
    # #TODO Change all weights according to total Truss Weight
    # for ii in range(n_frames):
    #     TotalWegiht += gamma*L*A_actual[ii] # *L[ii]
    # if any(d >= 30*m/360 for d in np.abs(uz_13)):
        print("uz_13 exceeded 30*m/360")
        Weights.append(10000000)
    else:
        Weights.append(TotalWegiht)
    # u = np.array(u)
    Weights = np.array(Weights)[0]
    # print(Weights)
    print('Weights:',Weights)
    return Weights/kN


# =============================================================================
#                                   GA
# =============================================================================
# Genetic Algorithm Part
# from geneticalgorithm import geneticalgorithm as ga

from geneticalgorithm2 import geneticalgorithm2 as ga
from geneticalgorithm2 import plot_several_lines

# Choose Particles
chromosome = 10
# Choose Iterations
iterations = 2

#TODO NOTE: make sure your SAP2000 model has units of "Tonf, m, C"
#TODO NOTE: lb and "Tonf" while calculating weights needs revision

n_frames = 21 # or 41
# MAXIMUM and MINIMUM #TODO (change this)
A1 = 50000.0/m**2 # mm**2
A2 = 600.0/m**2 # mm**2

# instatiate the optimizer #TODO (change this)
x_max = A1 * np.ones(n_frames)
x_min = A2 * np.ones(n_frames)
varbound=np.array([[A2,A1]]*n_frames)

# instatiate the optimizer
algorithm_param = {'max_num_iteration': iterations,\
                   'population_size': chromosome,\
                   'mutation_probability': 0.2,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.2,\
                   'parents_portion': 0.2,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':100}

# Define the model
model=ga(function=truss,\
            dimension=n_frames,\
            variable_type='real',\
            variable_boundaries=varbound,\
            algorithm_parameters=algorithm_param,
            function_timeout = 1300) 

# Run the GA model
results = model.run()
# Get the best Truss Weight 
print("Model parameters are:",algorithm_param)
print("Final Weight is:","{:.2f}".format(round(results.best_function, 3)),"kN")
print("Best cross-sections are:",results.variable*m**2,"mm**2")

#plot reports
names = [name for name, _ in model.checked_reports[::-1]]
plot_several_lines(
    lines=[getattr(model, name) for name in names],
    colors=('green', 'black', 'red', 'blue'),
    labels=['median value', '25% quantile', 'mean of population', 'best pop score'],
    linewidths=(1, 1.5, 1, 2),
    title="Several custom reports with base reports",
    save_as='./output/report.png'
)

pos_Names = []
df = pd.read_csv('Runs_GA/TUBO.csv')
for ij in range(n_frames):
    Area = results.variable[ij]
    temp = df[df['Area'] >= Area].iloc[0]
    Design_Section = temp['SectionName']
    pos_Names.append(Design_Section)
print("Best Sections are:",pos_Names)

# Total Optimization time
bb = datetime.datetime.now()
cc = bb - aa
print("The whole optimization process lasted:",cc,"hrs:mins:seconds")



# Get optimal

# get its name, place it in F3
with open(filename1) as f:
    lines = f.readlines()
    a = SearchKeyword('FRAME SECTION ASSIGNMENTS', lines)
    if a == -1:
        print('Frame Sections Definition NOT Found')
    else:
        frame_total = 0
        for ij,line in enumerate(lines[a:a+n_frames]):
            # print(ij)
            Area_Required = pos_Names[ij]
            if not line.isspace():
                result = line.split()
                F1 = result[0].split('=')[1]
                F2 = result[1].split('=')[1]
                F3 = result[2].split('=')[1]
                F4 = result[3].split('=')[1]
                
                # Modify Sections to Best
                
                F3 = Area_Required
                
                print(F3)
                
                lines[a+ij] = '   Frame='+str(F1)+'   AutoSelect='+str(F2)+\
                    '   AnalSect='+str(F3)+'   MatProp='+str(F4)+'\n'
                frame_total += 1
            else:
                print('Frame Sections Assignments have Been Updated')
                break
        print("All frames assigned, they are:", frame_total)
            

# Open the output file for writing
with open(filename3, "w") as output_file:
    # Write the new contents to the output file
    output_file.writelines(lines)
    print('Frame Sections Assignments have Been Written')