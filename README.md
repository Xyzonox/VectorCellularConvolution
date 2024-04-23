# Vector Cellular Convolution 
Simple program in Python Taichi that uses cell convolution for cellular automation running on the GPU. Convolution kernel is variable, mapped to how many cells a particular cell can "see" around it in all directions, where this program uses a Moore neighborhood. However, instead of each color channel of a cell being influenced by only the associate color channel of neighbor cells (which would essentially make 3 layers of separate Cellular Convolution environments), a Selection of the color channels from neighbor cells, using a Select function: Average, Maximum, Minimum, forms a matrix of dimension (2 x cellVision-1)x(2 x cellVision-1) which is then convoluted with the Kernel. The sum of the resulting matrix is applied to an Activation Function for each color channel. Of course, with a Select functions a cell can easily have more than three channels, where this program can assign an arbitrary amount of channels to each cell, though each activation function clamps the first three channels (which work as the pixel color channels) between 0 and 1, and each proceeding channel by some range scaled by its index. 

Each cell has the dimension n (Red, Green, Blue, (n-3) Channels), the Kernel for convolution has the dimensions (2 x cellVision-1) x (2 x cellVision-1) x n, where (2 xcellVision-1) represents the vertical or horizontal number of cells in a neighborhood (including the center cell) depending on how far it can see (cellVision) in any direction. After the channel selection the neighbor matrix has the dimensions (2 x cellVision-1)x(2 x cellVision-1), where convolution takes place for each layer of the Kernel. If the Kernel is devoid of noise between the channels, the result of the convolution will be monochromatic (as each color channel will have the same value), though with noise patterns are more complex as the Select functions have more information to process.

# Requirements
Python>= 3.10.14

Taichi>= 1.7.0

```
pip install taichi 
```

# Controls
Left Mouse Click: iterate through random KernelFactors and starting cell states (As long as randfac isn't zero)

R Key: Press once to start recording and again to finish recording a snapshot, all snapshots are cut togther upon program exit (Clear generated \GlobalSession Directory for new videos)

# Samples
![video](https://github.com/Xyzonox/VectorCellularConvolution/assets/85287832/2d91deb9-1ea8-4367-b2ec-ef1689393628)

(Video is very compressed) 

