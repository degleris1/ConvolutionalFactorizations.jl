using ConvolutionalFactorizations
using LinearAlgebra

# Test tensor convolution
N, T, K, L = 49, 101, 3, 11

W = rand(K, N, L)
H = rand(K, T)
B = tensor_conv(W, H)

@show ((N, T), size(B))
@assert (N, T) == size(B)


# Fit a model
model = ConvolutionalFactorization(L=L, K=K)
(W, H), cache, report = fit(model, B; verbosity=1)

B̂ = tensor_conv(W, H)
@show norm(B̂ - B)