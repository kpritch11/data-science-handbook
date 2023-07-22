#!/usr/bin/env python
# coding: utf-8

# # 2 NumPy

# No matter what the data are, the first step in making them analyzable will be to transform them into arrays of numbers.

# For this reason, efficient storage and manipulation of numerical arrays is absolutely fundamental to the process of doing data science. 
# <br>We’ll now take a look at the specialized tools that Python has for handling such numerical arrays: the NumPy package and the Pandas package .

import numpy as np
np.__version__


# ## 2.1 Understanding Data Types in Python

# Python's type flexibility also points to the fact that python variables are more than just their value.
# <br>They also contain extra information about the type of the value.

# ### A Python Integer is More Than Just an Integer

# ### A Python List is More Than Just a List

# list of integers
L = list(range(10))
L


# type of items in list
type(L[0])


# list of strings
L2 = [str(c) for c in L]
L2


# type of items in list
type(L2[0])


# heterogeneous list
L3 = [True, "2", 3.0, 4]
[type(item) for item in L3]


# At the implementation level, the array essentially contains a single pointer to one contiguous block of data. 
# <br>The Python list, on the other hand, contains a pointer to a block of pointers, each of which in turn points to a full Python object like the Python integer we saw earlier. 
# <br>Again, the advantage of the list is flexibility: because each list element is a full structure containing both data and type information, the list can befilled with data of any desired type. 
# <br>Fixed-type NumPy-style arrays lack this flexibility, but are much more efficient for storing and manipulating data.

# ### Fixed Type Arrays in Python

# built-in array module can be used to create dense arrays of a uniform type
import array
L = list(range(10))
A = array.array('i', L)
A


# Much more useful, however, is the ndarray object of the NumPy package. 
# <br>While Python’s array object provides efficient storage of array-based data, NumPy adds to this efficient operations on that data.

# ### Creating Arrays from Python Lists

# integer array
np.array([1, 4, 2, 5, 3])


# np will upcast integers to floating point if there is a mix of types
np.array([3.14, 4, 2, 3])


# explicitly set the data type
np.array([1, 2, 3, 4], dtype='float32')


# nested lists result in multidimensional arrays
np.array([range(i, i + 3) for i in [2, 4, 6]])


# ### Creating Arrays from Scratch

# create a length-10 integer array filled with zeros
np.zeros(10, dtype=int)


# create a 3x5 floating-point array filled with 1s
np.ones((3, 5), dtype=float)


# create a 3x5 array filled with 3.14
np.full((3, 5), 3.14)


# create an array filled with a linear sequence
# starting at 0, ending at 20, stepping by 2
# (this is similar to the built-in range() function)
np.arange(0, 20, 2)


# create an array of five values evenly spaced between 0 and 1
np.linspace(0, 1, 5)


# create a 3x3 array of uniformly distributed random valuex
# between 0 and 1
np.random.random((3, 3))


# create a 3x3 array of normally distributed random values
# with mean 0 and standard deviation 1
np.random.normal(0, 1, (3, 3))


# create a 3x3 array of random integers in the interval [0, 10)
np.random.randint(0, 10, (3, 3))


# create a 3x3 identity matrix
np.eye(3)


# create an uninitialized array of three integers
# the values will be whatever happens to already exist at that
# memory location
np.empty(3)


# ### NumPy Standard Data Types

# ## 2.2 The Basics of NumPy Arrays

# ### NumPy Array Attributes

# three random arrays: one-dimensional, two-dimensional, three-dimensional
import numpy as np
np.random.seed(0) # seed for reproducibility

x1 = np.random.randint(10, size=6) # one-dimensional array
x2 = np.random.randint(10, size=(3, 4)) # two-dimensional array
x3 = np.random.randint(10, size=(3, 4, 5)) # three-dimensional array


# number of dimensions of the array
# shape (size of each dimension) of the array
# size (total size) of the array
print("x3 ndim: ", x3.ndim)
print("x3 shape:", x3.shape)
print("x3 size: ", x3.size)


# the data type of the array
print("dtype:", x3.dtype)


# the size (in bytes) of each array element
# the total size (in bytes) of the array
print("itemsize:", x3.itemsize, "bytes")
print("nbytes:", x3.nbytes, "bytes")
# in general, we expect that nbytes is equal to itemsize times size


# ### Array Indexing: Accessing Single Elements

# one-dimensional array
x1


# one-dimensional array, access the ith value (counting from zero)
x1[0]


x1[4]


# index from the end of the array with negative indices
x1[-1]


x1[-2]


# multidiemensional array, access items using a comma-separated tuple of indices
x2


x2[0, 0]


x2[2, 0]


x2[2, -1]


# modify values 
x2[0, 0] = 12
x2


# insert floating point value to an integer array, the value will be truncated
x1[0] = 3.14159 # this will be truncated
x1


# ### Array Slicing: Accessing Subarrays

# array slices: x[start:stop:step]

