# Alternating optimization

abstract type AbstractUpdateRule end

function fit_alternating(
    model::ConvolutionalFactorization
    B,
    W0,
    H0;
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
    # b, loss_func, penalizers, constraint, set_gradx!, predict!, compute_loss
    # TODO define set_gradx!, predict!, and compute_loss
    M1 = (B, model.loss, model.W_penalizers, model.W_constraint)
    M2 = (B, model.loss, model.H_penalizers, model.H_constraints)

    submodels = [M1, M2]
    variables = [(B̂, W, H), (B̂, H, W)]

    optimizers = [RULES[model.algorithm], RULES[model.algorithm]]
    opt_states = initialize.(optimizers, variables, submodels)

    # Set up optimization tracking
    loss_hist = [compute_loss(model, B, W, H)]
    time_hist = [0.0]
    t0 = time()

    for itr in 1:model.max_itr 
        verbose && print(".")

        # Update
        for i in eachindex(opt_states)
            loss, state = step!(optimizers[i], opt_states[i])
            opt_states[i] = state

            # Record time and loss
            push!(time_hist, time() - t0)
            push!(loss_hist, loss)
        end

        # Check convergence
        (time_hist[end] >= model.max_time) && break
        (check_convergence && converged(loss_hist, patience, tol)) && break
    end

    verbose && println(" fit!")

    return (W, H), nothing, (time_hist=time_hist, loss_hist=loss_hist)
end

function compute_loss(model::ConvolutionalFactorization, X, W, H)
    loss = model.loss(X, tensor_conv(W, H))
    loss += sum([R(W) for R in model.W_penalizers])
    loss += sum([R(H) for R in model.H_penalizers])
    return loss
end


"""
Compute the gradient with respect to W -- dW J(W) = C(H)^T db̂.
"""
function get_gradW!(dW, vars)
    W, H, b̂ = vars

    # Unpack dimensions
    K, N, L, T = dims

    # Compute the transpose convolution, C(H)^T r
    for lag = 0:(L-1)
        @views mul!(dW[:, :, lag+1], shift_cols(H, lag), est[:, 1+lag:T]')
    end

    return dW
end