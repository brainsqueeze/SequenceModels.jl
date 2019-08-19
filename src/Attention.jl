import Flux

include("TensorOps.jl")

struct Attention{T}
    W::Flux.TrackedArray{T, 2}
    B::Flux.TrackedArray{T, 2}
    U::Flux.TrackedArray{T, 2}
end

Attention(dims::Integer) = Attention(
    Flux.param(Flux.gpu(randn(Float32, dims, dims))),
    Flux.param(Flux.gpu(zeros(Float32, 1, dims))),
    Flux.param(Flux.gpu(zeros(Float32, 1, dims))))

function (m::Attention)(x::AbstractArray{T, 3} where T)
    # x is the encoded input, the channels are (T, D, N)
    # this is equivalent to einsum("jmi,mk,->jki", x, W) + B
    logit = TensorDot(x, m.W) .+ m.B

    score = tanh.(logit)
    score = sum(m.U .* score, dims=2)
    α = TensorSoftmax(score, dims=2)
    return dropdims(sum(x .* α, dims=1), dims=1)
end

function (m::Attention)(x::AbstractArray{T, 3}, y::AbstractArray{T, 3}) where T
    # x is the encoded input, y the decoded input, the channels are (T, D, N)
    # this is equivalent to einsum("jmi,mn,kni->jki", x, W, y), with softmax on each batch matrix
    score = BatchMatMul(TensorDot(x, m.W), permutedims(y, [2, 1, 3]))
    α = TensorSoftmax(score, dims=2)
    α = sum(α, dims=1)
    return dropdims(sum(y .* permutedims(α, [2, 1, 3]), dims=1), dims=1)
end
Flux.@treelike Attention
