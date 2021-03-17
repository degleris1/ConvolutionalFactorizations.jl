# MLJ Model

@mlj_model mutable struct ConvolutionalFactorization <: Unsupervised
    L::Int = 1::(_ > 0)
    K::Int = 1::(_ > 0)
    loss::AbstractLoss = SquareLoss()
    W_penalizers::Vector{AbstractPenalty} = AbstractPenalty[]
    H_penalizers::Vector{AbstractPenalty} = AbstractPenalty[]
    W_constraint::AbstractConstraint = NoConstraint()
    H_constraint::AbstractConstraint = NoConstraint()
    algorithm::Symbol = :pgd::(_ in (:pgd,))
    max_iters::Int = 100::(_ > 0)
    max_time::Float64 = Inf::(_ > 0)
end

function MLJModelInterface.fit(
    model::ConvolutionalFactorization,
    X;
    verbosity::Int = 0,
    seed::Int = 123,
)
    return fit(model, verbosity, X; seed=seed)
end

function MLJModelInterface.fit(
    model::ConvolutionalFactorization, 
    verbosity::Int, 
    X;
    seed::Int = 123 
)
    (seed !== nothing) && seed!(seed)

    # Initialize
    W0, H0 = init_rand(X, model.L, model.K)

    # Fit with alternating optimization
    return fit_alternating(model, X, W0, H0, verbose=(verbosity > 0))
end

function MLJModelInterface.update(
    model::ConvolutionalFactorization, 
    verbosity::Int,
    old_fitresult,
    old_cache,
    X
)
    return old_fitresult, old_cache, nothing
end

function MLJModelInterface.transform(
    model::ConvolutionalFactorization,
    fitresult,
    Xnew;
    seed=123
)
    return tensor_conv(fitresult.W, fitresult.H)
end




# ===
# HELPER FUNCTIONS
# ===

function init_rand(B, L, K)
    N, T = size(B)

    W = rand(K, N, L)
    H = rand(K, T)

    B̂ = tensor_conv(W, H)
    α = (reshape(B, N*T)' * reshape(B̂, N*T)) / norm(B̂)^2
    W *= sqrt(abs(α))
    H *= sqrt(abs(α))

    return W, H
end