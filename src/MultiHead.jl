using Flux
using LinearAlgebra

include("TensorOps.jl")

struct MultiHeadAttention{T}
    WQueries::AbstractArray{T, 1}
    WKeys::AbstractArray{T, 1}
    WValues::AbstractArray{T, 1}
    HeadDense::Flux.TrackedArray
    L::Integer
end

function MultiHeadAttention(dims::Integer, layers::Integer)
    OutDims = Int32(floor(dims / layers))
    MultiHeadAttention(
        [Flux.param(randn(Float32, dims, OutDims)) for _ in 1:layers],
        [Flux.param(randn(Float32, dims, OutDims)) for _ in 1:layers],
        [Flux.param(randn(Float32, dims, OutDims)) for _ in 1:layers],
        Flux.param(randn(Float32, dims, dims)),
        layers
    )
end
Flux.@treelike MultiHeadAttention

function ScalarDotAttention(Query::AbstractArray{T, 3}, Key::AbstractArray{T, 3}, Value::AbstractArray{T, 3}; futuremask = false) where T
    Numerator = cat([@inbounds Query[:, :, batch] * transpose(Key[:, :, batch]) for batch in 1:size(Query, 3)]..., dims=3)
    Denominator = sqrt(size(Key, 1))

    if futuremask
        (S, D, N) = size(Numerator)
        Mask = ones(Float32, S, D)
        # lower triangular matrix with very large negative numbers
        Mask = LinearAlgebra.tril!(trues(size(Mask)), -1) * Float32(-1e9)
        Mask = Flux.param(Mask) |> Flux.gpu
        Numerator = Numerator .* Mask
    end

    x = TensorSoftmax(Numerator / Denominator, dims=2)
    return cat([@inbounds x[:, :, batch] * Value[:, :, batch] for batch in 1:size(Query, 3)]..., dims=3)
end

function _MHALayer(MHA::MultiHeadAttention, MHAData::AbstractArray{T, 1} where T, L::Integer; futuremask = false)
    (Query, Key, Value) = MHAData
    Nbatches = size(Query, 3)

    HQueries = cat([@inbounds Query[:, :, batch] * MHA.WQueries[L] for batch in 1:Nbatches]..., dims=3)
    HKeys = cat([@inbounds Key[:, :, batch] * MHA.WKeys[L] for batch in 1:Nbatches]..., dims=3)
    HValues = cat([@inbounds Value[:, :, batch] * MHA.WValues[L] for batch in 1:Nbatches]..., dims=3)

    return ScalarDotAttention(HQueries, HKeys, HValues, futuremask=futuremask)
end

function (m::MultiHeadAttention)(Query::AbstractArray{T, 3}, Key::AbstractArray{T, 3}, Value::AbstractArray{T, 3}; layers = 1, futuremask = false) where T
    TotalHeads = cat([_MHALayer(m, [Query, Key, Value], layer, futuremask=futuremask) for layer in 1:m.L]..., dims=2)
    return cat([@inbounds TotalHeads[:, :, batch] * m.HeadDense[:, :] for batch in 1:size(Query, 3)]..., dims=3)
end
