# Vector Cellular Convolution 
Simple program in Python Taichi that uses cell convolution for cellular automation running on the GPU. Convolution kernel is variable, mapped to how many cells a particular cell can "see" around it in all directions. However, instead of each color channel of a cell being influenced by only the associate color channel of neighbor cells (which would essentially make 3 layers of separate Cellular Convolution environments), a Selection of the color channels from neighbor cells, using a Select function: Average, Maximum, Minimum, forms a matrix of dimension (2 x cellVision-1)x(2 x cellVision-1) which is then convoluted with the Kernel and the sum is applied to an Activation Function for each color channel.

Each cell has the dimension 3 (Red, Green, Blue), the Kernel for convolution has the dimensions (2 x cellVision-1) x (2 x cellVision-1)x3, where (2 xcellVision-1) represents the neighbors of a cell depending on how far it can see in any direction (including diagonal), cellVision. After the channel selection the neighbor matrix has the dimensions (2 x cellVision-1)x(2 x cellVision-1), where convolution takes place for each layer of the Kernel. If the Kernel is devoid of noise between the color channels, the result of the convolution will be monochromatic (as each color channel will have the same value), though with noise patterns are more complex as the Select functions have more information to process.

# Requirements
Python= 3.10
Taichi

# Samples

![Screenshot 2024-04-20 064144](https://github.com/Xyzonox/VectorCellularConvolution/assets/85287832/bda59719-c020-4a8c-93c8-4c4d56aa200e)

![Screenshot 2024-04-20 063944](https://github.com/Xyzonox/VectorCellularConvolution/assets/85287832/1b3247e7-a59f-4a1d-983c-df96097ef05b)

![Screenshot 2024-04-20 064050](https://github.com/Xyzonox/VectorCellularConvolution/assets/85287832/de9d0ff7-b7d5-42de-8439-a038aa51d3ad)

![Screenshot 2024-04-20 064200](https://github.com/Xyzonox/VectorCellularConvolution/assets/85287832/e084412f-2e66-4861-95be-7c6ee8dd27ea)
