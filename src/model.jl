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

function compute_loss(model::ConvolutionalFactorization, X, W, H)
    loss = model.loss(X, tensor_conv(W, H))
    loss += sum([R(W) for R in model.W_penalizers])
    loss += sum([R(H) for R in model.H_penalizers])
    return loss
end

function MLJModelInterface.fit(
    model::ConvolutionalFactorization, 
    verbosity::Int, 
    X;
    seed::Int = 123 
)
    (seed !== nothing) && Random.seed!(seed)

    # Initialize
    W0, H0 = init_rand(X, model.L, model.K)

    # Fit with alternating optimization
    W, H, report = fit_alternating(model, X, W0, H0, verbose=(verbosity > 0))

    fitresult = (W=W, H=H)
    cache = nothing
    return fitresult, cache, report
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

function init_rand(data, L, K)
    N, T = size(data)

    W = rand(K, N, L)
    H = rand(K, T)

    est = tensor_conv(W, H)
    alpha = (reshape(data, N*T)' * reshape(est, N*T)) / norm(est)^2
    W *= sqrt(abs(alpha))
    H *= sqrt(abs(alpha))

    return W, H
end