import taichi as ti 
import numpy as np

@ti.func
def Sigmoid(cell:float, channel:float):
    result = cell
    if channel <= 2:
        result = (1/(1+ ti.math.e**(-1*cell))) #Sigmoid
    else:
        result = (channel/(1+ ti.math.e**(-1*cell)))
    return result

@ti.func
def RevGauss(cell:float, channel:float):
    result = cell
    if int(channel) <= 2:
        result = (1 + -1*ti.math.e**(-1*(cell**2)/2)) #Reverse Gaussian
    else:
        result = (channel + -1 * channel * ti.math.e**(-1*(cell**2)/2)) - channel/2
    return result

@ti.func
def Squiggle(cell:float, channel:float):
    result = cell
    if cell > 2:
        result = cell -1
    elif cell > 1:
        result = (cell/2)
    elif cell < -1:
        result = 1 + cell
    elif cell <= 0:
        result = cell + (-.9) * cell

    if channel <= 2:
        result = clamp(result, 1, 0)
    else:
        result = clamp(result, channel, -1 * channel)
    
    return result
@ti.func
def clamp(x:float, upper:float, lower:float):
    result = x
    if x > upper:
        result = upper
    elif x < lower:
        result = lower
    return result
@ti.func
def AvNeighbors (cell:float, channel:float, filterSize:float):
    result = cell/(filterSize * filterSize)
    if channel <= 2:
        result = clamp(result, 1, 0)
    else:
        result = clamp(result, channel, -1* channel)
    return result

@ti.func
def Selector(select:int, cell:float, channel:float, filterSize:float, upper:float, lower:float):
    # Taichi doesn't allow non-scalars in its dicitonaries, 
    # and Taichi expressions do not worth with python dictionaries
    # so a giant decision tree is inevitable
    result = 1.0
    if select == 1:
        result = Squiggle(cell, channel)
    elif select == 2:
        result = RevGauss(cell, channel)
    elif select == 3:
        result = Sigmoid(cell, channel)
    elif select == 4:
        result = clamp(cell, upper, lower)
    elif select == 5:
        result = AvNeighbors(cell, channel, filterSize)
    else:
        result = Squiggle(cell, channel)
    return result