# Alternating optimization

abstract type AbstractUpdateRule end

function fit_alternating(
    model::ConvolutionalFactorization,
    B, W0, H0;
    verbose=false,
    check_convergence=true,
    patience=3,
    eval_mode=false,
    tol=1e-4,
    kwargs...
)
    @assert patience >= 1

    # Initialize matrices
    W = deepcopy(W0)
    H = deepcopy(H0)
    B̂ = tensor_conv(W, H)

    verbose && print("Starting ")

    # Initialize stuff
    L = model.loss
    get_loss() = compute_loss(model, B, B̂, W, H)
    sync!() = tensor_conv!(B̂, W, H)
    
    R_W, C_W = model.W_penalizers, model.W_constraint
    set_dW!(dW) = get_dW!(dW, B̂, H)
    
    R_H, C_H = model.H_penalizers, model.H_constraint
    set_dH!(dH) = get_dH!(dH, B̂, W)

    θ1 = (B, L, R_W, C_W, set_dW!, sync!, get_loss)
    θ2 = (B, L, R_H, C_H, set_dH!, sync!, get_loss)

    submodels = [θ1, θ2]
    variables = [(B̂, W), (B̂, H)]

    optimizers = [RULES[model.algorithm], RULES[model.algorithm]]
    opt_states = initialize.(optimizers, variables, submodels)

    # Set up optimization tracking
    loss_hist = [compute_loss(model, B, W, H)]
    time_hist = [0.0]
    t0 = time()

    for itr in 1:model.max_iters
        verbose && print(".")

        # Update
        for i in eachindex(opt_states)
            state = step!(optimizers[i], opt_states[i])
            opt_states[i] = state

            # Record time and loss
            push!(time_hist, time() - t0)
            push!(loss_hist, state.loss)
        end

        # Check convergence
        (time_hist[end] >= model.max_time) && break
        # (check_convergence && converged(loss_hist, patience, tol)) && break
    end

    verbose && println(" fit!")

    return (W=W, H=H), nothing, (time_hist=time_hist, loss_hist=loss_hist)
end

function compute_loss(model::ConvolutionalFactorization, B, B̂, W, H)
    loss = model.loss(B, B̂)
    loss += sum(Float64[R(W) for R in model.W_penalizers])
    loss += sum(Float64[R(H) for R in model.H_penalizers])
    return loss
end

compute_loss(model::ConvolutionalFactorization, B, W, H) = 
    compute_loss(model, B, tensor_conv(W, H), W, H)

function get_dW!(dW, B̂, H)
    # Unpack dimensions
    K, N, L = size(dW)
    T = size(H, 2)

    # Compute the transpose convolution, C(H)^T r
    @inbounds for lag = 0:(L-1)
        @views mul!(dW[:, :, lag+1], shift_cols(H, lag), B̂[:, 1+lag:T]')
    end

    return dW
end

function get_dH!(dH, B̂, W)
    # Compute the transpose convolution, ∇ = C(W)^T r
    return tensor_transconv!(dH, W, B̂)
end