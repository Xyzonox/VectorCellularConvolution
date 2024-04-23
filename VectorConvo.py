import taichi as ti 
import numpy as np
import random
import argparse

parser = argparse.ArgumentParser(prog="VectorConvo", description="Placeholder", epilog="Placeholder")
parser.add_argument("--Resolution", help="Int; How many pixels in a square GUI")
parser.add_argument("--CellVision", help="Int; How far each cell can see all aroud itself")
parser.add_argument("--Pattern", help="Int; Kernel for Matrix Convolution: -1: Custom Pattern (Specified using --CustomFilterKernel), 0:Random, 1:Pattern1, 2:Pattern2, 3:Pattern3")
parser.add_argument("--KernelFactor", help="Float; How large elements in Kernel get, make it lower for exploding cells and higher for vanishing cells")
parser.add_argument("--RandomFactor", help="Float; Amount of randomness in Kernel, 0 for monochromatic cells")
parser.add_argument("--SelectFunc", help="Int; Function that decides what color channels to apply convolution to: 1:Average, 2:Minimum, 3:Maximum, 4:FlipFlop, 5:ChannelLength")
parser.add_argument("--ActFunc", help="Int; Activation function for each cell: 1:Squiggle, 2:RevGauss, 3:Sigmoid, 4:Clamp, 5:AvNeighbors")
parser.add_argument("--StartGrid", help="Int; What the starting grid looks like: 0:WhiteNoise, 1:Grid, 2:R-G Coordinates")
parser.add_argument("--Seed", help="Int; Seed for deterministic random value generation")
parser.add_argument("--CellDimension", help="Int; How many parameters a cell has and will be seen by other cell, minimum is 3")
parser.add_argument("--CustomFilterKernel", help="PATH; use a custom filter kernel specified in csv file. Should be of size (2*CellVision+1)x(2*CellVision+1)")


args = parser.parse_args()

def main(): 
    params = Arguments() #[seed, n, kernelfac,randfac,Pattern,cellVision, select, activation, start, cellDimension, customFilterKernel]
    ti.init(arch=ti.cuda, random_seed=params[0])
    filterSize = int(2*params[5]+1)
    iteration = 0

    global filterKernel
    global filterRandom
    global cells
    global pixels
    global prev
    global prevPixel
    global convolv
    global accum

    filterKernel = ti.Vector.field(params[9],dtype=float, shape=(filterSize,filterSize))
    filterRandom = ti.Vector.field(params[9],dtype=float, shape=(filterSize,filterSize))
    cells = ti.Vector.field(params[9],dtype=float, shape=(params[1],params[1])) #Stores values per pixel to be displayed
    pixels = ti.Vector.field(3,dtype=float, shape=(params[1],params[1])) #Stores values per cell 
    prev = ti.Vector.field(params[9],dtype=float, shape=(params[1],params[1])) #Stores values per pixeled for last iteration
    prevPixel = ti.Vector.field(3,dtype=float, shape=(params[1],params[1]))
    convolv = ti.Vector.field(params[9],dtype=float, shape=(params[1],params[1])) #Stores convolution data for every pixel
    accum = ti.Vector.field(3,dtype=float, shape=(params[1],params[1])) #Stores values per pixel to be displayed, For Blur
    
    Info(params[0], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9])
    gui = ti.GUI("Cell Auto", res=(params[1],params[1]), fast_gui=True)
    setup(params[2], params[3], params[4], filterSize, params[8], params[9], params[10])
    gui.set_image(prevPixel)
    gui.show()
    record = False
    video_manager = ti.tools.VideoManager(output_dir="Outputs/Output"+str(iteration), framerate=24, automatic_build=False)
    while gui.running:
        for e in gui.get_events(gui.PRESS):
            if e.key == ti.GUI.LMB:
                iteration += 1
                setup(params[2], params[3], params[4], filterSize, params[8], params[9], params[10])
                print("Generation Iteration: ", iteration)
            elif e.key == "r":
                if not(record):
                    record = True
                else:
                    record = False
                    video_manager.make_video(gif=True)
        if record:
            video_manager.write_frame(pixels)
        CellAuto(params[5], filterSize, params[6], params[7], params[9])
        paint(params[5], filterSize, params[9])
        gui.set_image(pixels)
        gui.show()
        prev.copy_from(cells)

def Arguments():
    if args.Seed:
        seed = int(args.Seed)
    else:
        seed = int(0)
    iteration = 0

    if args.Resolution:
        n= int(args.Resolution) or int(300)
    else:
        n = int(500)

    if args.KernelFactor: #Multiple for convolution kernel, Turn this up if cells are vanishing, down if blowing up / flickering
        kernelfac = float(args.KernelFactor) 
    else:
        kernelfac =  1. 
    
    if args.RandomFactor: #The amount of randomness each element in filterKernel has
        randfac = float(args.RandomFactor) 
    else: 
        randfac = .1
    
    if args.Pattern:
        Pattern = int(args.Pattern) 
    else: 
        Pattern = int(0)
    
    if args.CellVision: #How many cells a cell can see in all directions around it
        cellVision = int(args.CellVision)
    else: 
        cellVision = int(4) 
    
    if args.SelectFunc:
        select = int(args.SelectFunc)
    else:
        select = int(5)
    
    if args.ActFunc:
        activation = int(args.ActFunc)
    else:
        activation = int(2)

    if args.StartGrid:
        start = int(args.StartGrid)
    else:
        start = int(0)

    if args.CellDimension:
        if int(args.CellDimension) >= 3:
            cellDimension = int(args.CellDimension)
        else:
            cellDimension = int(3)
    else:
        cellDimension = int(6)

    if args.CustomFilterKernel:
        customFilterKernel = (np.genfromtxt(args.CustomFilterKernel, dtype=np.single, delimiter=","))
    else:
        customFilterKernel = (np.ndarray(shape=(2,2),dtype=np.single))
    params = [seed, n, kernelfac,randfac,Pattern,cellVision, select, activation, start, cellDimension, customFilterKernel]
    return params

def Info(seed, kernelfac, randfac, Pattern, cellVision, select, activation, start, cellDimension):
    #print(filterKernel)
    print("Seed: ", seed)
    print("Cell Vision: ", cellVision)
    print("Kernel Factor: ", kernelfac)
    print("Random Factor: ", randfac)
    print("Cell Dimention: ", cellDimension)
    if Pattern == 0:
        print("Filter Pattern: Random")
    elif Pattern < 0:
        print("Used Custom KernelFilter")
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
    
    if start == 1:
        print("Starting Pattern: Grid")
    else:
        print("Starting Pattern: White Noise")

#Activation Functions, Pixel Space
@ti.func
def Sigmoid(x: float, k):
    result = x
    if k <= 2:
        result = (1/(1+ ti.math.e**(-1*x))) #Sigmoid
    else:
        result = (k/(1+ ti.math.e**(-1*x)))
    return result
@ti.func
def RevGauss(x: float, k:float):
    result = x
    if k <= 2:
        result = (1+ -1*ti.math.e**(-1*(x**2)/2)) #Reverse Gaussian
    else:
        result = (k+ -1*k*ti.math.e**(-1*(x**2)/2)) -k/2
    return result
@ti.func
def Squiggle(x: float, k):
    result = x
    if x > 2:
        result = x-1
    elif x > 1:
        result = (x/2)
    elif x < -1:
        result = 1+x
    elif x <= 0:
        result = x + (-.9) * x

    if k <= 2:
        result = clamp(result, 1, 0)
    else:
        result = clamp(result, k, -1 * k)
    
    return result
@ti.func
def clamp(x: float, upper, lower):
    result = x
    if x > upper:
        result = upper
    elif x < lower:
        result = lower
    return result
@ti.func
def AvNeighbors (x:float, filterSize, k):
    result = x/(filterSize * filterSize)
    if k <= 2:
        result = clamp(result, 1, 0)
    else:
        result = clamp(result, k, -1* k)
    return result

#Channel Select Functions, Vector Space
@ti.func
def Minim(x, CellDimension:ti.i16):
    index = x[0]
    for k in range(CellDimension):
        if x[k] < index:
            index = x[k]
    return index
@ti.func
def Maxim(x, CellDimension:ti.i16):
    index = x[0]
    for k in range(CellDimension):
        if x[k] > index:
            index = x[k]
    return index
@ti.func
def Average(x, CellDimension:float):
    collect = 0.
    for k in range(ti.i16(CellDimension)):
        collect += x[k]
    result = collect/CellDimension
    return result
@ti.func
def FlipFlop(x, CellDimension:ti.i16):
    collect = 0.
    for k in range(ti.i16(CellDimension)):
        if k/2 == int(k/2):
            collect += x[k]
        else:
            collect -= x[k]
    #result = collect/CellDimension
    return collect
@ti.func
def ChannelLength(x):
    return ti.math.length(x)

#Procedural FilterKernel Generation
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
def setup(kernelfac:ti.f32, randfac:ti.f32, Pattern:ti.i16, filterSize:ti.i16, start:ti.i16, CellDimension:ti.i16, customFilterKernel:ti.types.ndarray(dtype=ti.f32)):
    for i, j in filterRandom:
        for k in range(CellDimension):
            filterRandom[i,j][k] = ((2*ti.random(float))-1) * randfac
    for d,e in filterKernel: #Kernel Matrix Generation, The goal is to have each elment in filterKernel to equal close to zero
        for k in range(CellDimension):
            if Pattern < 0:
                filterKernel[d,e][k] = customFilterKernel[d,e] + filterRandom[d,e][k]
            elif Pattern == 1:
                filterKernel[d,e][k] = Pattern1(d,e,filterSize,kernelfac) + filterRandom[d,e][k]
            elif Pattern == 2:
                filterKernel[d,e][k] = Pattern2(d,e,filterSize,kernelfac) + filterRandom[d,e][k]
            elif Pattern == 3:
                filterKernel[d,e][k] = Pattern3(d,e,filterSize,kernelfac) + filterRandom[d,e][k]
            else:
                filterKernel[d,e][k] = ((filterRandom[d,e][k])) * kernelfac #Random Matrix
    for i, j in prev: #Starting Pixel State
        if start == 1:
            prev[i,j] = clamp((i % -50)+5, 1, 0) + clamp((j % -50)+5, 1, 0) #Grid
        else:
            for k in range(CellDimension):
                prev[i,j][k] = ti.random(float) # white noise
            for k in range(3):
                prevPixel[i,j][k] = prev[i,j][k]
@ti.kernel
def CellAuto(cellVision:ti.i16, filterSize:ti.i16, select:ti.i16, activation:ti.i16, CellDimension:ti.f32): #Cell Automata
    for i, j in pixels: #parallized over pixels
        for k in range(ti.i16(CellDimension)):
            convolv[i,j][k] = 0

        for dx, dy in ti.ndrange(filterSize,filterSize):
            row = (dx + i - cellVision) % ((prev.shape[0]))
            col = (dy + j - cellVision) % ((prev.shape[1]))
            for k in range(ti.i16(CellDimension)):
                #Select Functions
                if select == 1:
                    convolv[i,j][k] += (Average(prev[row,col], CellDimension) * filterKernel[dx,dy][k])
                elif select == 2:
                    convolv[i,j][k] += (Minim(prev[row,col], CellDimension) * filterKernel[dx,dy][k])
                elif select == 3:
                    convolv[i,j][k] += (Maxim(prev[row,col], CellDimension) * filterKernel[dx,dy][k])
                elif select == 4:
                    convolv[i,j][k] += (FlipFlop(prev[row,col], CellDimension) * filterKernel[dx,dy][k])
                elif select== 5:
                    convolv[i,j][k] += (ChannelLength(prev[row,col]) * filterKernel[dx,dy][k])
                else:
                    convolv[i,j][k] += (Average(prev[row,col], CellDimension) * filterKernel[dx,dy][k])

        for k in range(ti.i16(CellDimension)):
            #Activation Functions
            if activation == 1:
                cells[i,j][k] = Squiggle(convolv[i,j][k], k)
            elif activation == 2:
                cells[i,j][k] = RevGauss(convolv[i,j][k], k)
            elif activation == 3:
                cells[i,j][k] = Sigmoid(convolv[i,j][k], k)
            elif activation == 4:
                cells[i,j][k] = clamp(convolv[i,j][k], 1, 0)
            elif activation == 5:
                cells[i,j][k] = AvNeighbors(convolv[i,j][k], filterSize, k)
            else:
                cells[i,j][k] = Squiggle(convolv[i,j][k], k)
@ti.kernel
def paint(cellVision:ti.i16, filterSize:ti.i16, CellDimension:ti.i16): #Post Processing
    for i, j in pixels: #parallized over pixels
        accum[i,j] = ti.Vector([0,0,0])
        count = int(0)
        for dx, dy in ti.ndrange(5,5):
            row = (dx + i - cellVision) % ((prev.shape[0]))
            col = (dy + j - cellVision) % ((prev.shape[1]))
            accum[i,j] += ti.Vector([cells[row, col][0], cells[row, col][1], cells[row, col][2]])
            count += 1

        pixels[i,j]= accum[i,j]/count


if __name__ == "__main__":
    main()
