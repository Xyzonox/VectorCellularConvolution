import taichi as ti 
import numpy as np

@ti.func
def Minim(channels, cellDimension):
    index = channels[0]
    for k in range(ti.i16(cellDimension)):
        if channels[k] < index:
            index = channels[k]
    return index

@ti.func
def Maxim(channels, cellDimension):
    index = channels[0]
    for k in range(ti.i16(cellDimension)):
        if channels[k] > index:
            index = channels[k]
    return index

@ti.func
def Average(channels, cellDimension):
    collect = 0.
    for k in range(ti.i16(cellDimension)):
        collect += channels[k]
    result = collect/cellDimension
    return result

@ti.func
def FlipFlop(channels:float, cellDimension:float):
    collect = 0.0
    for k in range(ti.i16(cellDimension)):
        if k/2 == int(k/2):
            collect += channels[k]
        else:
            collect -= channels[k]
    return collect

@ti.func
def ChannelLength(channels, cellDimension):
    return ti.math.length(channels)

@ti.func
def ComplexDive(channels, cellDimension, i, j, res):
    real = (channels[1] + channels[2] + channels[3])
    comp = 0.
    for k in range(ti.i16(cellDimension - 3)):
        comp += channels[k+3]
    c = ti.Vector([real, comp])
    z = ti.Vector([i/res - 1, j/res - 0.5]) * 2
    iterations = 0
    while z.norm() < 5 and iterations < 10:
        z = ti.math.cpow(z, 0.5) + c
        iterations += 1
    return iterations


@ti.func
def Selector(select:int, channels:float, cellDimension:float, i:float, j:float, res:float):
    # Taichi doesn't allow non-scalars in its dicitonaries, 
    # and Taichi expressions do not worth with python dictionaries
    # so a giant decision tree is inevitable
    result = 1.0
    if select == 1:
        result = Average(channels, cellDimension)
    elif select == 2:
        result = Minim(channels, cellDimension)
    elif select == 3:
        result = Maxim(channels, cellDimension)
    elif select == 4:
        result = FlipFlop(channels, cellDimension)
    elif select == 5:
        result = ChannelLength(channels, cellDimension)
    elif select == 6:
        result = ComplexDive(channels, cellDimension, i, j, res)
    else:
        result = Average(channels, cellDimension)
    return result