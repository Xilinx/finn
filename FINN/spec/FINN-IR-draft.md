# Draft for FINN intermediate representation

This draft describes the layer types that appear in the FINN intermediate representation, and what properties/parameters they have.

General notes:
* The FINN IR is not intended for "general" neural networks, so it's better to make things as restrictive as possible when in doubt. These restrictions may be relaxed as support for more general computation is implemented.
* The IR should contain both "high level" primitives (like a convolution layer) and the building blocks they can map down to (pad-sliding window-matrix matrix).
* Set "consume" and "produce" for each node (except first and last) to allow arbitrary graph structure, like Caffe, using either buffer names or layer names to connect.

One possibility for serialization:
* JSON for network architecture. Each layer is a node with unique name. We should consider making a schema for validation.
* NPZ for parameter storage. Match layers in JSON to params in NPZ using unique name plus field name (e.g. the weights W of a layer conv0 would be conv0_W).

Primitive types used in the representation:
* String
* NumberType = bipolar, int[1-8], uint[1-8], int16, uint16, int32, uint32, float
* ImageShape = (channels: uint32, height: uint32, width: uint32)
* NumberArray = NumPy array

## InputLayer
Describes the shape of the data input to the network (number of channels and dimension), precision in bits per element, and whether it uses interleaved channels. 
* img: (shape: ImageShape, dtype: NumberType). Required. Shape and data type of the image.
* description: String. Optional. A human-readable description of what kind of images the network accepts, and how they must be preprocessed.

## MatrixVectorLayer
A layer that multiplies its input vector x with a matrix W. The input will be flattened prior to multiplication and must match the number of matrix columns. If bias is desired, it must be implemented as a LinearLayer following this layer.
* in: (shape: ImageShape, dtype: NumberType). Required. Shape and data type of the input image. The product of the elements of the shape = col.
* out: (shape: ImageShape, dtype: NumberType). Required. Shape and data type of the output image. The product of the elements of the shape = row.
* W: (data: NumberArray, dtype: NumberType). Required. 2D weight matrix for the fully connected layer, stored in row-major format W[row][col].
* acctype: NumberType. Required. Type of accumulator variable used in the computation.

## ConvolutionLayer
A layer that implements a convolution layer that connects each of its input channels to each of its output channels with a different kernel. If padding is desired, it must be implemented as a PaddingLayer preciding this layer. If bias is desired, it must be implemented as a LinearLayer following this layer. Output dimension of images:  ((input dimension - conv_kernel_dim) / stride) + 1
* in: (shape: ImageShape, dtype: NumberType). Required. Shape and data type of the input image. Number of channels is IFM.
* out: (shape: ImageShape, dtype: NumberType). Required. Shape and data type of the output image. Number of channels is OFM.
* W: (data: NumberArray, dtype: NumberType). Required. 4D weight tensor for the convolutions, organized as W[OFM][IFM][Y][X].
* stride: (ystride: uint32, xstride: uint32). Required. Stride along y and x dimensions for applying the convolution.
* acctype: NumberType. Required. Type of accumulator variable used in the computation.

## SlidingWindowLayer
TODO

## MatrixMatrixLayer
TODO

## QuantizationLayer
TODO

## PaddingLayer
Pads the images with desired value. Output dimension of images: input dimension + 2*pad (in each axis).
* in: (shape: ImageShape, dtype: NumberType). Required. Shape and data type of the input image. 
* out: (shape: ImageShape, dtype: NumberType). Required. Shape and data type of the output image. 
* pad_amount: (ypad: uint32, xpad: uint32). Required. Number of elements to add on each border for padding.
* pad_value: <same dtype as in and out>. Required. The value used for padding.

## ThresholdLayer
Returns the number of crossed (>=) thresholds. The input will be flattened prior to applying the linear transformation. If the threshold channels is smaller than the flattened input vector, the threshold vectors will be repeated to match the size of the input vector. Input and output shapes are the same. The output data type from the layer is uint8.
* img: (shape: ImageShape, dtype: NumberType). Required. Shape and data type of the input image. 
* num_threshold_levels: uint32. Required. Number of threshold levels (equal to number of output levels minus one).
* T: (data: NumberArray, dtype: NumberType). Required. Threshold matrix, organized as T[levels][channels]. dtype should be the same as the input dtype.

## LinearLayer
A layer that applies a linear transformation Ax + B on its input x. The input will be flattened prior to applying the linear transformation. A and B can either be scalars or 1D vectors whose size divides the size of the input vector. If A and B are smaller than the flattened input vector, they will be repeated to match the size of the input vector. Most or all instances of this layer should be absorbed into thresholds for networks using uniform quantization as the activation. Input and output shapes and data types are the same.
* img: (ImageShape, NumberType). Required. Shape and data type of the image.
* A: NumberArray. Required. Parameter to scale (multiply) the input with.
* B: NumberArray. Required. Parameter to add to the scaled input.

## PoolingLayer
Pooling. Output dimension of images:  ((input dimension - pool_dim) / stride) + 1
* pooltype: String. Required. Either MAX or AVG.
* in: (shape: ImageShape, dtype: NumberType). Required. Shape and data type of the input image. 
* out: (shape: ImageShape, dtype: NumberType). Required. Shape and data type of the output image.
* pool_region: (pooly: uint32, poolx: uint32). Required. Size of pooling region.
* stride: (ystride: uint32, xstride: uint32). Required. Stride along y and x dimensions for applying the convolution.

## OutputLayer
Describes the shape of the data output from the network (number of channels and dimension), precision in bits per element. 
* img: (shape: ImageShape, dtype: NumberType). Required. Shape and data type of the image.
* description: String. Optional. A human-readable description of what kind of output the network produces.
