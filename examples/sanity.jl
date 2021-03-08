using ConvolutionalFactorizations

# Test tensor convolution
N, T, K, L = 49, 101, 3, 11

W = rand(K, N, L)
H = rand(K, T)

@show ((N, T), size(tensor_conv(W, H)))
@assert (N, T) == size(tensor_conv(W, H))