import taichi as ti 
import taichi.math as math
import numpy as np
import random
import argparse

parser = argparse.ArgumentParser(prog="VectorConvo", description="Placeholder", epilog="Placeholder")
parser.add_argument("--Resolution", help="Int; How many pixels in a square GUI")
parser.add_argument("--CellVision", help="Int; How far each cell can see all aroud itself")
parser.add_argument("--Pattern", help="Int; Kernel for Matrix Convolution: 0:Random, 1:Pattern1, 2:Pattern2, 3:Pattern3")
parser.add_argument("--KernelFactor", help="Float; How large elements in Kernel get, make it lower for exploding cells and higher for vanishing cells")
parser.add_argument("--RandomFactor", help="Float; Amount of randomness in Kernel, 0 for monochromatic cells")
parser.add_argument("--SelectFunc", help="Int; Function that decides what color channels to apply convolution to: 1:Average, 2:Minimum, 3:Maximum")
parser.add_argument("--ActFunc", help="Int; Activation function for each cell: 1:Squiggle, 2:RevGauss, 3:Sigmoid, 4:Clamp")
parser.add_argument("--StartGrid", help="Int; What the starting grid looks like: 0:WhiteNoise, 1:Grid, 2:R-G Coordinates")
parser.add_argument("--Seed", help="Int; Seed for deterministic random value generation")

args = parser.parse_args()

#ti.init(arch=ti.cuda, random_seed=1)

def main(): 
    if args.Seed:
        seed = int(args.Seed)
    else:
        seed = int(0)
    iteration = 0

    if args.Resolution:
        n= int(args.Resolution) or int(300)
    else:
        n = int(300)

    if args.KernelFactor: #Multiple for convolution kernel, Turn this up if cells are vanishing, down if blowing up / flickering
        kernelfac = float(args.KernelFactor) 
    else:
        kernelfac =  1. 
    
    if args.RandomFactor: #The amount of randomness each element in filterKernel has
        randfac = float(args.RandomFactor) 
    else: 
        randfac = 1.
    
    if args.Pattern:
        Pattern = int(args.Pattern) 
    else: 
        Pattern = int(0)
    
    if args.CellVision: #How many cells a cell can see in all directions around it
        cellVision = int(args.CellVision)
    else: 
        cellVision = int(2) 
    
    if args.SelectFunc:
        select = int(args.SelectFunc)
    else:
        select = int(1)
    
    if args.ActFunc:
        activation = int(args.ActFunc)
    else:
        activation = int(1)

    if args.StartGrid:
        start = int(args.StartGrid)
    else:
        start = int(0)

    ti.init(arch=ti.cuda, random_seed=seed)
    filterSize = int(2*cellVision+1)

    global filterKernel
    global filterRandom
    global cells
    global pixels
    global prev
    global convolv
    global accum
    filterKernel = ti.Vector.field(3,dtype=float, shape=(filterSize,filterSize))
    filterRandom = ti.Vector.field(3,dtype=float, shape=(filterSize,filterSize))
    cells = ti.Vector.field(3,dtype=float, shape=(n,n)) #Stores values per pixel to be displayed
    pixels = ti.Vector.field(3,dtype=float, shape=(n,n)) #Stores values per cell 
    prev = ti.Vector.field(3,dtype=float, shape=(n,n)) #Stores values per pixeled for last iteration
    convolv = ti.Vector.field(3,dtype=float, shape=(n,n)) #Stores convolution data for every pixel
    accum = ti.Vector.field(3,dtype=float, shape=(n,n)) #Stores values per pixel to be displayed, For Blur
    
    Info(cellVision, kernelfac, randfac, Pattern, select, activation)
    gui = ti.GUI("Cell Auto", res=(n,n))
    setup(kernelfac, randfac, Pattern, filterSize, start)
    gui.set_image(prev)
    gui.show()
    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            iteration += 1
            setup(kernelfac, randfac, Pattern, filterSize, start)
            print("Generation Iteration: ", iteration)
        CellAuto(cellVision, filterSize, select, activation)
        paint(cellVision, filterSize)
        gui.set_image(pixels)
        gui.show()
        prev.copy_from(cells)

def Info(cellVision, kernelfac, randfac, Pattern, select, activation):
    #print(filterKernel)
    print("Cell Vision: ", cellVision)
    print("Kernel Factor: ", kernelfac)
    print("Random Factor: ", randfac)
    if Pattern == 0:
        print("Filter Pattern: Random")
    else:
        print("Filter Pattern: ", Pattern) 

    if select == 1:
        print("Select Function: Average")
    elif select == 2:
        print("Select Function: Minimum")
    elif select == 3:
        print("Select Function: Maximum")
    
    if activation == 1:
        print("Activation Function: Squiggle")
    elif activation == 2:
        print("Activation Function: Reverse Gaussian")
    elif activation == 3:
        print("Activation Function: Sigmoid")
    elif activation == 4:
        print("Activation Function: Clamp (0 and 1) ")

#Activation Functions, Pixel Space
@ti.func
def Sigmoid(x: float):
    result = (1/(1+ ti.math.e**(-1*x))) #Sigmoid
    return result
@ti.func
def RevGauss(x: float):
    result = (1+ -1*ti.math.e**(-1*(x**2)/2)) #Reverse Gaussian
    return result
@ti.func
def Squiggle(x: float):
    result = x
    if x > 2:
        result = 1
    elif x > 1:
        result = (x/2)
    elif x < -1:
        result = 1+x
    elif x <= 0:
        result = x + (-.9) * x
    return clamp(result)
@ti.func
def clamp(x: float):
    result = x
    if x > 1:
        result = 1
    elif x < 0:
        result = 0
    return result

#Channel Select Functions, Vector Space
@ti.func
def Minim(x):
    return ti.min(x[0], x[1], x[2])
@ti.func
def Maxim(x):
    return ti.max(x[0], x[1], x[2])
@ti.func
def Average(x):
    return (x[0] + x[1] + x[2])/3

#FilterKernel Generation
@ti.func
def Pattern1(d:float, e:float, filterSize:float, kernelfac:float): #Works better CellVision > 2 at base kernelfac
    row = d
    row = (row - (filterSize-1)/2)
    if row < 0:
        row = row * -1
    row = (row/2)

    col = e
    col = (col - (filterSize-1)/2)
    if col < 0:
        col = col * -1
    col = (col/2)

    result = (kernelfac * ((row + col)/((filterSize-1)/4) - .75*(filterSize/7)))
    Pattern = 1
    return result
@ti.func
def Pattern2(d:float, e:float, filterSize:float, kernelfac:float): #Works better CellVision > 1 at base kernelfac
    row = d
    row = (row - (filterSize-1)/2)
    if row < 0:
        row = row * -1
    row = (row - (filterSize-1)/2)
    if row < 0:
        row = row * -1

    col = e
    col = (col - (filterSize-1)/2)
    if col < 0:
        col = col * -1
    col = (col - (filterSize-1)/2)
    if col < 0:
        col = col * -1
    result = (kernelfac * (((row + col - 1))/(filterSize-2) - 0.27*(filterSize/5)))
    Pattern = 2
    return result
@ti.func
def Pattern3(d:float, e:float, filterSize:float, kernelfac:float):
    row = d
    row = (row - (filterSize-1)/2)

    col = e
    col = (col - (filterSize-1)/2) 
    if col > 0:
        col = row
    if row > 0:
        row = col

    result = ((row + col + 1))
    Pattern = 3
    return result

#Kernels
@ti.kernel
def setup(kernelfac:ti.f32, randfac:ti.f32, Pattern:ti.i16, filterSize:ti.i16, start:ti.i16):
    for i, j in filterRandom:
        for k in range(3):
            filterRandom[i,j][k] = ((2*ti.random(float))-1) * randfac
    for d,e in filterKernel: #Kernel Matrix Generation, The goal is to have each elment in filterKernel to equal close to zero
        for k in range(3): 
            if Pattern == 1:
                filterKernel[d,e][k] = Pattern1(d,e,filterSize,kernelfac) + filterRandom[d,e][k]
            elif Pattern == 2:
                filterKernel[d,e][k] = Pattern2(d,e,filterSize,kernelfac) + filterRandom[d,e][k]
            elif Pattern == 3:
                filterKernel[d,e][k] = Pattern3(d,e,filterSize,kernelfac) + filterRandom[d,e][k]
            else:
                filterKernel[d,e][k] = ((filterRandom[d,e][k])/randfac) * kernelfac #Random Matrix
    for i, j in prev: #Starting Pixel State
        if start == 1:
            prev[i,j] = clamp((i % -50)+5) + clamp((j % -50)+5) #Grid
        else:
            prev[i,j] = ti.Vector([ti.random(float), ti.random(float), ti.random(float)]) # white noise
@ti.kernel
def CellAuto(cellVision:ti.i16, filterSize:ti.i16, select:int, activation:int): #Cell Automata
    for i, j in pixels: #parallized over pixels
        convolv[i,j] = ti.Vector([0,0,0])
        for dx, dy in ti.ndrange(filterSize,filterSize):
            row = (dx + i - cellVision) % ((prev.shape[0]))
            col = (dy + j - cellVision) % ((prev.shape[1]))
            for k in range(3):
                #Vector Space
                if select == 1:
                    convolv[i,j][k] += (Average(prev[row,col]) * filterKernel[dx,dy][k])
                elif select == 2:
                    convolv[i,j][k] += (Minim(prev[row,col]) * filterKernel[dx,dy][k])
                elif select == 3:
                    convolv[i,j][k] += (Maxim(prev[row,col]) * filterKernel[dx,dy][k])
                else:
                    convolv[i,j][k] += (Average(prev[row,col]) * filterKernel[dx,dy][k])
        
        for k in range(3):
            #Pixel Space
            if activation == 1:
                cells[i,j][k] = Squiggle(convolv[i,j][k])
            elif activation == 2:
                cells[i,j][k] = RevGauss(convolv[i,j][k])
            elif activation == 3:
                cells[i,j][k] = Sigmoid(convolv[i,j][k])
            elif activation == 4:
                cells[i,j][k] = clamp(convolv[i,j][k])
            else:
                cells[i,j][k] = Squiggle(convolv[i,j][k])
@ti.kernel
def paint(cellVision:ti.i16, filterSize:ti.i16): #Post Processing
    for i, j in pixels: #parallized over pixels
        accum[i,j] = ti.Vector([0,0,0])
        count = int(0)
        for dx, dy in ti.ndrange(filterSize,filterSize):
            row = (dx + i - cellVision) % ((prev.shape[0]))
            col = (dy + j - cellVision) % ((prev.shape[1]))
            accum[i,j] += cells[row, col]
            count += 1

        pixels[i,j]= accum[i,j]/count


if __name__ == "__main__":
    main()