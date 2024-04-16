import taichi as ti 
import taichi.math as math
import random

#Parameters
ti.init(arch=ti.gpu)
n=800 # resolution
kernelfac = 1 #Multiple for convolution kernel, Turn this up if cells are vanishing, down if blowing up / flickering
cellVision = 2 #How many cells a cell can see in all directions around it

filterSize = 2*cellVision+1
filterKernel = ti.Vector.field(3,dtype=ti.f32, shape=(filterSize,filterSize))
filterRandom = ti.Vector.field(3,dtype=ti.f32, shape=(filterSize,filterSize))
cells = ti.Vector.field(3,dtype=ti.f32, shape=(n,n)) #Stores values per pixel to be displayed
pixels = ti.Vector.field(3,dtype=ti.f32, shape=(n,n)) #Stores values per cell 
prev = ti.Vector.field(3,dtype=ti.f32, shape=(n,n)) #Stores values per pixeled for last iteration
convolv = ti.Vector.field(3,dtype=ti.f32, shape=(n,n)) #Stores convolution data for every pixel
accum = ti.Vector.field(3,dtype=ti.f32, shape=(n,n)) #Stores values per pixel to be displayed, For Blur
 
#Non deterministic number generation
for i in range(filterSize):
    for j in range(filterSize):
        for k in range(3):
            filterRandom[i,j][k] = (2*random.random())-1

@ti.func
def rand():
    return ti.random(ti.f32)


#Activation Functions
@ti.func
def activate(x: float):
    result = 0
    #result = 1/(1+ ti.math.e**(-1*x)) #Sigmoid
    result = 1+ -1*ti.math.e**(-1*(x**2)/2) #Reverse Gaussian
    #if x > 1:
        #result = 1
    #elif x < 0:
        #result = 0
    return result
@ti.func
def Sigmoid(x: float):
    result = 1/(1+ ti.math.e**(-1*x)) #Sigmoid
    return result
@ti.func
def RevGauss(x: float):
    result = 1+ -1*ti.math.e**(-1*(x**2)/2) #Reverse Gaussian
    return result
@ti.func
def Squiggle(x: float):
    result = x
    if x > 2:
        result = 1
    elif x > 1:
        result = x/2
    elif x < -1:
        result = 1+x
    elif x <= 0:
        result = x + -.9*x
    return clamp(result)
@ti.func
def clamp(x: float):
    result = x
    if x > 1:
        result = 1
    elif x < 0:
        result = 0
    return result

#Channel Select Functions
@ti.func
def Minim(x):
    return ti.min(x[0], x[1], x[2])
@ti.func
def Maxim(x):
    return ti.max(x[0], x[1], x[2])
@ti.func
def Average(x):
    return (x[0] + x[1] + x[2])/3

#Kernel Matrix Values
@ti.func
def Pattern1(d, e, filterSize, kernelfac):
    row = d
    row = (row - (filterSize-1)/2)
    #if row < 0:
        #row = row * -1

    col = e
    col = (col - (filterSize-1)/2)
    #if col < 0:
        #col = col * -1

    result = kernelfac * ((row + col - 1))
    return result
@ti.func
def Pattern2(d, e, filterSize, kernelfac):
    row = d
    row = (row - (filterSize-1)/2)
    if row < 0:
        row = row * -1
    row = row - (filterSize-1)/2
    if row < 0:
        row = row * -1
    #row = row/2

    col = e
    col = (col - (filterSize-1)/2)
    if col < 0:
        col = col * -1
    col = col - (filterSize-1)/2
    if col < 0:
        col = col * -1
    result = kernelfac * ((row + col - 1))
    return result

@ti.func
def Pattern3(d, e, filterSize, kernelfac):
    row = d
    row = (row - (filterSize-1)/2)

    col = e
    col = (col - (filterSize-1)/2) 
    if col > 0:
        col = row
    if row > 0:
        row = col

    result = ((row + col + 1))
    return result

@ti.kernel
def setup():
    for d,e in filterKernel: #Kernel Matrix Generation
        for k in range(3): 
            #filterKernel[d,e][k] = ((rand()*2)-1) * kernelfac #Random Matrix
            filterKernel[d,e][k] = Pattern3(d,e,filterSize,kernelfac) + filterRandom[i,j][k] * kernelfac 
    for i, j in prev: #parallized over pixels
        #prev[i,j] = ti.Vector([rand(), rand(), rand()]) # white noise
        prev[i,j] = clamp((i % -50)+5) + clamp((j % -50)+5) #Grid

@ti.kernel
def CellAuto():
    for i, j in pixels: #parallized over pixels
        convolv[i,j] = ti.Vector([0,0,0])
        for dx, dy in ti.ndrange(filterSize,filterSize):
            row = (dx + i - cellVision) % ((prev.shape[0]))
            col = (dy + j - cellVision) % ((prev.shape[1]))
            for k in range(3):
                #convolv[i,j][k] += ti.min(prev[row, col][0], prev[row, col][1], prev[row, col][2]) * filterKernel[dx,dy][k] #* prev[row, col][k]
                convolv[i,j][k] += Average(prev[row,col]) * filterKernel[dx,dy][k]  
        
        for k in range(3):
            cells[i,j][k] = Squiggle(convolv[i,j][k])  

@ti.kernel
def paint():
    for i, j in pixels: #parallized over pixels
        accum[i,j] = ti.Vector([0,0,0])
        count = 0
        for dx, dy in ti.ndrange(filterSize,filterSize):
            row = (dx + i - cellVision) % ((prev.shape[0]))
            col = (dy + j - cellVision) % ((prev.shape[1]))
            accum[i,j] += cells[row, col]
            count += 1

        pixels[i,j]= accum[i,j]/count


gui = ti.GUI("Cell Auto", res=(n,n))

setup()
print(filterKernel)
gui.set_image(prev)
gui.show()
while gui.running:
    CellAuto()
    paint()
    gui.set_image(pixels)
    gui.show()
    prev.copy_from(cells)
