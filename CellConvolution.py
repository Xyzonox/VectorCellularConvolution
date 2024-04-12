import taichi as ti 
import taichi.math as math
import random

ti.init(arch=ti.gpu)
n=300 # resolution
pixels = ti.field(dtype=ti.f32, shape=(n,n)) #Stores values per pixel to be displayed
prev = ti.field(dtype=ti.f32, shape=(n,n)) #Stores values per pixeled for last iteration
convolv = ti.field(dtype=ti.f32, shape=(n,n)) #Stores convolution data for every pixel

kernelfac = 7 #Multiple for convolution kernel
cellVision = 2 #How many cells a cell can see in all directions around it
filterSize = 2*cellVision+1
filterKernel = ti.field(dtype=ti.f32, shape=(filterSize,filterSize))
 
@ti.func
def rand():
    return ti.random(ti.f32)

@ti.func
def activate(x: float):
    result = 0
    #result = 1/(1+ ti.math.e**(-1*x))
    result = 1+ -1*ti.math.e**(-1*(x**2)/2)
    if x > 1:
        result = 1
    elif x < 0:
        result = 0
    return result

@ti.kernel
def setup():
    for d,e in filterKernel:
        filterKernel[d,e] = ((rand()*2)-1) * kernelfac #Kernel Matrix Generation
    for i, j in prev: #parallized over pixels
        prev[i,j] = (rand())

@ti.kernel
def paint():
    for i, j in pixels: #parallized over pixels
        convolv[i,j] = 0
        for dx, dy in ti.ndrange(filterSize,filterSize):
            row = dx + i - cellVision % ((prev.shape[0]))
            col = dy + j - cellVision % ((prev.shape[1]))
            convolv[i,j] += prev[row, col] * filterKernel[dx,dy]
        pixels[i,j] = activate(convolv[i,j])  


gui = ti.GUI("Cell Auto", res=(n,n))
setup()
print(filterKernel)
while gui.running:
    paint()
    gui.set_image(pixels)
    gui.show()
    prev.copy_from(pixels)