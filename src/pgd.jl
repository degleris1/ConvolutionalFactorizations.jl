# Projected gradient descent

@kwdef struct ProjectGradientDescent <: AbstractUpdateRule
    step_incr
    step_decr
    step_init
end

ProjectGradientDescent() = 
    ProjectGradientDescent(step_incr=1.05, step_decr=0.70, step_init=5)


function initialize(opt::ProjectGradientDescent, vars, model)
    b̂, x, θ = vars

    η = opt.step_init
    dx = similar(x)

    return (η=η, dx=dx, loss=?, vars=vars, model=model)
end

function step!(opt::ProjectGradientDescent, state)
    η, dx, old_loss, vars, model = state

    b̂, x, θ = vars
    b, loss_func, penalizers, constraint, set_gradx!, predict!, compute_loss = model
    
    # Step 1: compute gradient (d/dx) J(x)
    # Compute (d/db̂) D(b, b̂) and store it in b̂
    grad!(loss_func, b̂, b̂, b)

    # Compute db̂/dx
    set_gradx!(dx, θ)
    dx .= 0  # TODO

    # Add in (d/dx) R(x)
    for R in penalizers
        grad!(R, dx, x)
    end

    # Step 2: update x and project
    α = η / (norm(dx) + eps())
    @. x -= α * dx
    projection!(constraint, x)

    # Step 3: update b̂ and step size
    predict!(b̂, x, θ) 
    new_loss = compute_loss(b, b̂, x, θ)
    η *= (new_loss < old_loss) ? opt.step_incr : opt.step_decr

    return (η=η, dx=dx, loss=new_loss, vars=vars, model=model)
end
