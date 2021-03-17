# Tensor-matrix convolutions

"""
    tensor_conv(W, H)

Computes the tensor convolution of `W` (with size `(N, K, L)`) and `H` (with size 
`(K, T)`), given by the formula:

    Σₗ W[:, :, l] ⊗ right_shift(H, l)

The output is a matrix of size `(N, T)`.
"""
function tensor_conv(W::AbstractArray, H::AbstractArray)
    K, N, L = size(W)
    est = zeros(N, size(H, 2))
    return tensor_conv!(est, W, H)
end

"""
    tensor_conv!(X, W, H)

Same as [`tensor_conv`](@ref), but saves memory by storing the output in `X` instead of
allocating new memory.
"""
function tensor_conv!(est, W::AbstractArray, H::AbstractArray)
    K, N, L = size(W)
    T = size(H, 2)

    @. est = 0
    for lag = 0:(L-1)
        @views _shift_dot!(est[:, lag+1:T], W[:, :, lag+1]', H, lag, 1, 1)
    end
    
    return est
end

"""
    _shift_dot!(B, Wl, H, lag, α, β)

Shifts the columns of `H` by `l` columns to the right, then computes 
`mul!(B, Wl, H_shifted, α, β)` (see `LinearAlgebra.mul!` for details.)
"""
function _shift_dot!(B, Wl, H, lag, α, β)
    K, T = size(H)
    
    if lag < 0
        @views mul!(B, Wl, H[:, 1+lag:T], α, β)
    else  # lag >= 0
        @views mul!(B, Wl, H[:, 1:T-lag], α, β)
    end

    return B
end

function tensor_transconv!(out, W, X)
    K, N, L = size(W)
    T = size(X, 2)

    @. out = 0
    for lag = 0:(L-1)
        @views mul!(out[:, 1:T-lag], W[:, :, lag+1], shift_cols(X, -lag), 1, 1)
    end

    return out
end

function shift_cols(X, lag)
    T = size(X, 2)
    
    if (lag <= 0)
        return view(X, :, 1-lag:T)

    else  # lag > 0
        return view(X, :, 1:T-lag)
    end
end