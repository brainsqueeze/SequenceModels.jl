using Flux

include("TensorOps.jl")

struct Attention
    W::Flux.TrackedArray{T, 2} where T
    B::Flux.TrackedArray{T, 2} where T
    U::Flux.TrackedArray{T, 2} where T
end

Attention(dims::Integer) = Attention(
    Flux.param(randn(Float32, dims, dims)),
    Flux.param(zeros(Float32, 1, dims)),
    Flux.param(zeros(Float32, 1, dims)))

function (m::Attention)(x::AbstractArray{T, 3} where T)
    # x is the encoded input, the channels are (T, D, N)
    # this is equivalent to einsum("jmi,mk,->jki", x, W) + B
    logit = [@inbounds x[:, :, batch] * m.W .+ m.B for batch in 1:size(x, 3)]
    logit = cat(logit..., dims=3)

    score = tanh.(logit)
    score = sum(m.U .* score, dims=2)
    # α = TensorSoftmax(score, dims=1)
    α = TensorSoftmax(score, dims=2)
    return sum(x .* α, dims=1)
end

function (m::Attention)(x::AbstractArray{T, 3} where T, y::AbstractArray{T, 3} where T)
    # x is the encoded input, y the decoded input, the channels are (T, D, N)
    # this is equivalent to einsum("jmi,mn,kni->jki", x, W, y), with softmax on each batch matrix
    α = [
        Flux.softmax(x[:, :, batch] * m.W * transpose(y[:, :, batch]))
        for batch in 1:size(x, 3)
    ]
    α = cat(α..., dims=3)
    α = sum(α, dims=1)
    return dropdims(sum(y .* permutedims(α, [2, 1, 3]), dims=1), dims=1)
end
Flux.@treelike Attention
