# Alternating optimization

abstract type AbstractUpdateRule end

RULES = Dict(
    :pgd => PGDUpdate,
)

function fit_alternating(
    model::ConvolutionalFactorization
    data,
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

    # Initialize matrices and update rule
    W = deepcopy(W0)
    H = deepcopy(H0)
    rule = RULES[model.algorithm](X, W0, H0)

    # Set up optimization tracking
    loss_hist = [evaluate_loss(model, W, H, data)]
    time_hist = [0.0]

    verbose && print("Starting ")

    itr = 1
    t0 = time()

    while (itr <= model.max_itr) && (time_hist[end] <= model.max_time) 
        itr += 1

        # Update
        update_motifs!(rule, model, data, W, H; kwargs...)
        loss = update_feature_maps!(rule, model, data, W, H; kwargs...)

        # Record time and loss
        push!(time_hist, time() - t0)
        push!(loss_hist, loss)
        
        verbose && print(".")

        # Check convergence
        if check_convergence && converged(loss_hist, patience, tol)
            println("Converged early.")
            break
        end
    end

    verbose && println(" fit!")

    return W, H, (time_hist=time_hist, loss_hist=loss_hist)
end