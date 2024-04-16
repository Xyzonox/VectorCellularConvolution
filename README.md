# miscellaneous Taichi Projects 
CellConvolution.py: Simple program in Python Taichi that uses cell convolution for cellular automation running on the GPU. Convolution kernel is variable, mapped to how many cell a particular cell can "see" around it in all directions

VectorCellConvolution.py: A similar premise to CellConvolution.py but expanded to RGB. However, instead of each color channel of a cell being influenced by only the associate color channel of neighbor cells (whuch would essentially make 3 layers of the results seen in CellConvolution.py), a Selection of the color channels from neighbor cells (Minimum, Maximum, Average) is applied to every channel of a cell during convolution. 

RayMarhcer.py: Simple ray marching script that renders a simple scene with basic diffuse and specular shading

# Requirements
Python= 3.10
Taichi
