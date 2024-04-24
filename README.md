# Vector Cellular Convolution 
A Simple program in Python Taichi Lang that uses cell convolution for cellular automation running on the GPU. Inspired by neuralpatterns.io by Emergent Garden. The difference with this is that (1) cells can see an arbitrary amount of other cells, (2) each cell can have as little as 3 channels (Red Green and Blue) and up to an arbitrary amount of channels (unseen higher dimensional channels). With the increased channels and arbitrary cell vision, Convolution kernel (also called Filter or FilterKernel) is completely variable, scaled to how many cells a particular cell can "see" around it in all directions and scaled to how many channels there are (creating a matrix of channel vectors), where this program uses a Moore neighborhood. 

Instead of each channel of a cell being influenced by only the associate channel of neighbor cells (where that would result in just different layers of separate cellular automaton that are completely independent of each other), a "Select Function" selects a value from the vector holding the values of each channel, for each cell in the neighborhood matrix. This determines the scalar value of a cell and its neighbors which then can be convolved with the Filter (for each channel of the filter, the Filter is still multidimentional). The sum of the resulting matrix is applied to an Activation Function for each channel, where the first 3 are clamped to 0-1 (as they are RGB channels), and the remaining have a more flexibile range of values (but are mostly scaled to their associate index in the channel vector). A Select function can be any function that turns a vector into a scalar. 

Each cell has the dimension 'n' containing [Red, Green, Blue, ('n'-3) high dimensional] Channels, the Filter for convolution has the dimensions (2 x cellVision-1) x (2 x cellVision-1) x 'n', where (2 x cellVision-1) represents the vertical or horizontal number of cells in a neighborhood (including the center cell) depending on how far it can see (cellVision) in any direction. After the channel selection the neighbor matrix has the dimensions (2 x cellVision-1)x(2 x cellVision-1), where convolution takes place for each layer of the Filter. If the Filter is devoid of noise between the channels, the result of the convolution will be monochromatic (as each channel of the Filter will have the same value by default, a more complex Filter generator and Filter reader needs to be implemented)

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

