# Projected gradient descent

@kwdef struct ProjectedGradientDescent <: AbstractUpdateRule
    step_incr
    step_decr
    step_init
end

pgd() = ProjectedGradientDescent(step_incr=1.05, step_decr=0.70, step_init=5)

function initialize(opt::ProjectedGradientDescent, vars, problem)
    b̂, x = vars
    b, loss_func, penalizers, constraint, set_dx!, sync!, compute_loss = problem

    η = opt.step_init
    dx = similar(x)
    loss = compute_loss()

    return (η=η, dx=dx, loss=loss, vars=vars, problem=problem)
end

function step!(opt::ProjectedGradientDescent, state)
    η, dx, old_loss, vars, problem = state

    b̂, x = vars
    b, loss_func, penalizers, constraint, set_dx!, sync!, compute_loss = problem
    
    # Step 1: compute gradient (d/dx) J(x)
    # Compute (d/db̂) D(b̂, b) and store it in b̂
    grad!(loss_func, b̂, b̂, b)

    # Compute db̂/dx
    set_dx!(dx)

    # Add in (d/dx) R(x)
    for R in penalizers
        grad!(R, dx, x)
    end

    # Step 2: update x and project
    α = η / (norm(dx) + eps())
    @. x -= α * dx
    projection!(constraint, x)

    # Step 3: update b̂ and step size
    sync!() 
    new_loss = compute_loss()
    η *= (new_loss < old_loss) ? opt.step_incr : opt.step_decr

    return (η=η, dx=dx, loss=new_loss, vars=vars, problem=problem)
end
