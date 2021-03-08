module ConvolutionalFactorizations

# ===
# IMPORTS
# ===

# Load modules
using LinearAlgebra: mul!
using MLJModelInterface: @mlj_model, Unsupervised

# Import modules for function redefinition
import MLJModelInterface




# ===
# EXPORTS
# ===

# Convolutions
export tensor_conv, tensor_conv!

# Loss functions, penalizers, and constraints
export SquareLoss, AbsoluteLoss, MaskedLoss
export SqaurePenalty, AbsolutePenalty
export NonnegConstraint, UnitNormConstraint




# ===
# TYPES AND CONSTANTS
# ===

notyetimplemented() = error("Not yet implemented.")

abstract type AbstractLoss end
abstract type AbstractPenalty end
abstract type AbstractConstraint end




# ===
# INCLUDE FILES
# ===

include("convolve.jl")





end
