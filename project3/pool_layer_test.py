from src.Layers import PoolingLayer
import numpy as np

### Sources for pooling layer dev:
### https://lanstonchu.wordpress.com/2018/09/01/convolutional-neural-network-cnn-backward-propagation-of-the-pooling-layers/

def test(size, typ):
    ## Simple FeedForward case, seeing if correct elements are chosen for max and calculated for mean
    x = np.arange(16).reshape(4,4)
    pool = PoolingLayer(x.shape, kernel_extent=size, pool_type=typ)
    _, a = pool(x)
    print(x)
    print(a)

    ## Making fake gradient for Backpropogation case, to see if it is distributed correctly
    dCda = np.zeros(pool.output_shape)
    dCda[0,0] = 1
    dCda[0,1] = 3
    dCda[1,0] = 0.7
    dCda[1,1] = 0.5

    dCda_new = pool.update_wb(dCda, None, None)
    print(dCda)
    print(dCda_new)

print("\nSize 2, max pooling:")
test(2, 'max')

print("\nSize 3, max pooling:")
test(3, 'max')

print("\nSize 2, mean pooling:")
test(2, 'mean')

print("\nSize 3, mean pooling:")
test(3, 'mean')