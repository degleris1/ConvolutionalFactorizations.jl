# Loss functions, penalizers, and constraints

abstract type AbstractLoss end
abstract type AbstractPenalty end
abstract type AbstractConstraint end

# ===
# LOSS FUNCTIONS
# ===

"""General loss function"""
AbstractLoss

"""
    grad!(D, grad, b̂, b)
    
Computes the gradient of `D(b̂, b)` and stores it in `grad`.
"""
grad!(D::AbstractLoss, grad, est, data) = notyetimplemented()

"""
    (D::AbstractLoss)(b̂, b)

Computes `D(b̂, b)`.
"""
(D::AbstractLoss)(b̂, b) = notyetimplemented()





"""
    D(b, b̂) = ||b - b̂||₂ ^ 2
"""
struct SquareLoss <: AbstractLoss end

function grad!(D::SquareLoss, grad, b_est, b)
    @. grad = 2 * (b_est - b)
    return grad
end

function (D::SquareLoss)(b_est, b)
    # Why not use norm(b_est - b)^2 here?
    # Because (b_est - b) will allocate memory and slow down performance

    # What about mapreduce((x,y) -> (x-y)^2, +, b_est, b)?
    # Something is up with the performance of mapreduce, so we're not going to use it.

    # This custom implementation uses constant memory and is relatively fast.

    total = 0.0
    @inbounds @simd for i in eachindex(b)
        total += (b_est[i] - b[i])^2
    end
    return total
end




"""
    D(b, b̂) = || b - b̂ ||₁
"""
struct AbsoluteLoss <: AbstractLoss end

function grad!(D::AbsoluteLoss, grad, b_est, b)
    @. grad = sign(b_est - b)
    return grad
end

function (D::AbsoluteLoss)(b_est, b)
    total = 0.0
    @inbounds @simd for i in eachindex(b)
        total += abs(b_est[i] - b[i])
    end
    return total
end




"""
    D(b, b̂) = loss(mask ⊗ b, mask ⊗ b̂)
"""
struct MaskedLoss <: AbstractLoss
    loss::AbstractLoss
    mask
end

(D::MaskedLoss)(b_est, b) = D.loss(D.mask .* b_est, D.mask .* b)

function grad!(D::MaskedLoss, grad, b_est, b)
    grad!(D.loss, grad, b_est, b)
    @. grad *= D.mask
    return grad
end





# ===
# PENALTIES
# ===

"""General penalizer."""
AbstractPenalty

weight(p::AbstractPenalty) = p.weight

"""
    grad!(P, grad, x)
    
Computes the gradient of `P(x)` and stores it in `grad`.
"""
grad!(P::AbstractPenalty, g, x) = notyetimplemented()

"""
    (P::AbstractPenalty)(x)

Computes `P(x)`.
"""
(P::AbstractPenalty)(x) = notyetimplemented()




"""
    R(x) = ||x||₂ ^ 2
"""
struct SquarePenalty <: AbstractPenalty
    weight
end

(P::SquarePenalty)(x) = P.weight * norm(x)^2

function grad!(P::SquarePenalty, g, x)
    @. g += 2 * P.weight * x
    return g
end


    

""" R(x) = ||x||_1 """
struct AbsolutePenalty <: AbstractPenalty
    weight
end

(P::AbsolutePenalty)(x) = P.weight * norm(x, 1)

function grad!(P::AbsolutePenalty, g, x)
    @. g += P.weight * sign(x)
    return g
end




# ===
# CONSTRAINTS
# ===

"""General constraint."""
AbstractConstraint

projection!(C::AbstractConstraint, x) = notyetimplemented()




"""
    No constraint.
"""
struct NoConstraint <: AbstractConstraint end

function projection!(C::NoConstraint, x)
    return x
end



"""
    ∀i, xᵢ ≥ 0
"""
struct NonnegConstraint <: AbstractConstraint end

function projection!(C::NonnegConstraint, x)
    @. x = max(eps(), x)
    return x
end




"""
    ||x||₂ ≤ 1
"""
struct UnitNormConstraint <: AbstractConstraint end

function projection!(C::UnitNormConstraint, x)
    M = size(x, 1)

    for m in 1:M
        xm = selectdim(x, 1, m)
        mag = norm(xm)
        if mag > 1
            @. xm /= mag
        end
    end

    return x
end
