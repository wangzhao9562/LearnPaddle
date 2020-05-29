# Numpy  
## ndarray  
### Create  
array - Receive a sequence type object  
arrange - Input a range of value and the step  
zeors - Create an 2 dimension array filled wtih 0 through list which defines the size  
ones - Create an 2 dimension array filled with 1 through list which defines the size  
### Attribute  
shape - Return the type of array  
dtype - Return the type of element    
size - Return the size fo vector  
ndim - Return the dimension of array  
astype - Change type of elements  
reshpe - Change shape of array  
### Compute  
scala * ndarray  
scala + ndarray  
scala / ndarray  
scala - ndarray  
ndarray - ndarray  
ndarray + ndarray  
ndarray * ndarray  
ndarray / ndarray  
### Slice and index  
slice - ndarray[-n,n-1]   
index - ndarray[i]  
slices = [ndarray[k:k+step] for k in range(start, stop, step)]  
### Statistic  
mean - Compute average of ordered axis, axis=1 represents the direction of row, axis=0 represents the direction of column  
std - Compute standard deviation of ordered axis   
var - Compute variance of ordered axis   
sum - Compute sum of ordered axis   
min - Get minimum element of ordered axis   
max - Get maximum element of ordered axis   
argmin - Get index of miminum element along ordered axis   
argmax - Get index of maximum element along ordered axis   
cumsum - Compute the sum of all elements  
cumprod - Compute the product of all elements  
### Random  
seed - set random seed  
uniform - Receive range and size, generates number through uniform distribution  
normal - Receive mean value, variance and size, generates number through normal distribution  
shuffle - Shuffle the elements of ndarray  
choice - Select an element randomly  
### Linear algebra  
diag - Return the diagnal elements of a matrix, or transform a vector into a matrix filled with only diagnal elements  
dot - Array multiplication rules  
trace - Sum of diagnal of matrix  
det - Compute determinant of matrix  
eig - Compute egien value and vector of matrix  
inv - Compute the inversion of matrix  
### File read and write  
fromfile - Receive the path of file and split char  
save - Save ndarray in received path  
load - Load ndarray from received path  





