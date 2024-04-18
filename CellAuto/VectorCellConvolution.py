import taichi as ti 
import taichi.math as math
import numpy as np
import random

#Parameters
rand = int(random.random()*100)
ti.init(arch=ti.cuda, random_seed=rand)
n=300 # resolution
kernelfac = 1 #Multiple for convolution kernel, Turn this up if cells are vanishing, down if blowing up / flickering
randfac = 0.1 #The amount of randomness each element in filterKernel has
cellVision = 3 #How many cells a cell can see in all directions around it

filterSize = 2*cellVision+1
filterKernel = ti.Vector.field(3,dtype=float, shape=(filterSize,filterSize))
filterRandom = ti.Vector.field(3,dtype=float, shape=(filterSize,filterSize))
cells = ti.Vector.field(3,dtype=float, shape=(n,n)) #Stores values per pixel to be displayed
pixels = ti.Vector.field(3,dtype=float, shape=(n,n)) #Stores values per cell 
prev = ti.Vector.field(3,dtype=float, shape=(n,n)) #Stores values per pixeled for last iteration
convolv = ti.Vector.field(3,dtype=float, shape=(n,n)) #Stores convolution data for every pixel
accum = ti.Vector.field(3,dtype=float, shape=(n,n)) #Stores values per pixel to be displayed, For Blur
 
Pattern = 0

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
def Pattern1(d, e, filterSize, kernelfac): #Works better CellVision > 2 at base kernelfac
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
def Pattern2(d, e, filterSize, kernelfac): #Works better CellVision > 1 at base kernelfac
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
def Pattern3(d, e, filterSize, kernelfac):
    row = d
    row = ti.float16(row - (filterSize-1)/2)

    col = e
    col = ti.float16(col - (filterSize-1)/2) 
    if col > 0:
        col = row
    if row > 0:
        row = col

    result = ((row + col + 1))
    Pattern = 3
    return result

#Kernels
@ti.kernel
def setup():
    for i, j in filterRandom:
        for k in range(3):
            filterRandom[i,j][k] = (2*ti.random(float))-1
    for d,e in filterKernel: #Kernel Matrix Generation, The goal is to have each elment in filterKernel to equal close to zero
        for k in range(3): 
            filterKernel[d,e][k] = filterRandom[d,e][k] * kernelfac #Random Matrix
            #filterKernel[d,e][k] = Pattern2(d,e,filterSize,kernelfac) + filterRandom[d,e][k] * randfac
    for i, j in prev: #Starting Pixel State
        prev[i,j] = ti.Vector([ti.random(float), ti.random(float), ti.random(float)]) # white noise
        #prev[i,j] = clamp((i % -50)+5) + clamp((j % -50)+5) #Grid
@ti.kernel
def CellAuto(): #Cell Automata
    for i, j in pixels: #parallized over pixels
        convolv[i,j] = ti.Vector([0,0,0])
        for dx, dy in ti.ndrange(filterSize,filterSize):
            row = (dx + i - cellVision) % ((prev.shape[0]))
            col = (dy + j - cellVision) % ((prev.shape[1]))
            for k in range(3):
                #Vector Space
                convolv[i,j][k] += (Minim(prev[row,col]) * filterKernel[dx,dy][k])
        
        for k in range(3):
            #Pixel Space
            cells[i,j][k] = Squiggle(convolv[i,j][k])  
@ti.kernel
def paint(): #Post Processing
    for i, j in pixels: #parallized over pixels
        accum[i,j] = ti.Vector([0,0,0])
        count = int(0)
        for dx, dy in ti.ndrange(filterSize,filterSize):
            row = (dx + i - cellVision) % ((prev.shape[0]))
            col = (dy + j - cellVision) % ((prev.shape[1]))
            accum[i,j] += cells[row, col]
            count += 1

        pixels[i,j]= accum[i,j]/count


gui = ti.GUI("Cell Auto", res=(n,n))
setup()
#print(filterKernel)
print("Seed: ", rand)
print("Cell Vision", cellVision)
print("Kernel Factor", kernelfac)
print("Random Factor", randfac)
if Pattern == 0:
    print("Filter Pattern: Random")
else:
    print("Filter Pattern: ", Pattern)
gui.set_image(prev)
gui.show()
while gui.running:
    CellAuto()
    paint()
    gui.set_image(pixels)
    gui.show()
    prev.copy_from(cells)
