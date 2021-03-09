module ConvolutionalFactorizations

# ===
# IMPORTS
# ===

# Load modules
using Base: @kwdef
using LinearAlgebra: mul!, norm
using MLJModelInterface: @mlj_model, Unsupervised, fit

# Import modules for function redefinition
import MLJModelInterface




# ===
# EXPORTS
# ===

# Convolutions
export tensor_conv, tensor_conv!

# Loss functions, penalizers, and constraints
export SquareLoss, AbsoluteLoss, MaskedLoss
export SquarePenalty, AbsolutePenalty
export NonnegConstraint, UnitNormConstraint

# Model
export ConvolutionalFactorization, fit




# ===
# INCLUDE FILES
# ===

# Atom types and functionality
include("atoms.jl")

# MLJ Model
include("model.jl")

# Convolution methods
include("convolve.jl")

# Algorithms
include("alternating.jl")
include("pgd.jl")




# ===
# TYPES AND CONSTANTS
# ===

notyetimplemented() = error("Not yet implemented.")

RULES = Dict(
    :pgd => ProjectedGradientDescent(),
)





end
