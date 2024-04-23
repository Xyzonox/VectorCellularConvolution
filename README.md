# Vector Cellular Convolution 
Simple program in Python Taichi that uses cell convolution for cellular automation running on the GPU. Convolution kernel is variable, mapped to how many cells a particular cell can "see" around it in all directions. However, instead of each color channel of a cell being influenced by only the associate color channel of neighbor cells (which would essentially make 3 layers of separate Cellular Convolution environments), a Selection of the color channels from neighbor cells, using a Select function: Average, Maximum, Minimum, forms a matrix of dimension (2 x cellVision-1)x(2 x cellVision-1) which is then convoluted with the Kernel and the sum is applied to an Activation Function for each color channel.

Each cell has the dimension 3 (Red, Green, Blue), the Kernel for convolution has the dimensions (2 x cellVision-1) x (2 x cellVision-1)x3, where (2 xcellVision-1) represents the neighbors of a cell depending on how far it can see in any direction (including diagonal), cellVision. After the channel selection the neighbor matrix has the dimensions (2 x cellVision-1)x(2 x cellVision-1), where convolution takes place for each layer of the Kernel. If the Kernel is devoid of noise between the color channels, the result of the convolution will be monochromatic (as each color channel will have the same value), though with noise patterns are more complex as the Select functions have more information to process.

# Requirements
Python>= 3.10.14

Taichi>= 1.7.0

# Controls
Left Mouse Click: iterate through random KernelFactors and starting cell states (As long as randfac isn't zero)

R Key: Press once to start recording and again to finish recording a snapshot, all snapshots are cut togther upon program exit

# Samples
![video](https://github.com/Xyzonox/VectorCellularConvolution/assets/85287832/2d91deb9-1ea8-4367-b2ec-ef1689393628)

(Video is very compressed) 

