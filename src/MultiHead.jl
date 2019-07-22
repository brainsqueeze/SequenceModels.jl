using Flux
using LinearAlgebra

include("TensorOps.jl")

struct MultiHeadAttention
    WQueries::AbstractArray{T, 1} where T
    WKeys::AbstractArray{T, 1} where T
    WValues::AbstractArray{T, 1} where T
    HeadDense::Flux.TrackedArray{T, 2} where T
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

function ScalarDotAttention(Query::AbstractArray{T, 3} where T, Key::AbstractArray{T, 3} where T, Value::AbstractArray{T, 3} where T; mask = false)
    Numerator = [@inbounds Query[:, :, batch] * transpose(Key[:, :, batch]) for batch in 1:size(Query, 3)]
    Numerator = cat(Numerator..., dims=3)
    Denominator = sqrt(size(Key, 1))

    if mask
        (T, D, N) = size(Numerator)
        Mask = ones(Float32, T, D, N)
        # lower triangular matrix with very large negative numbers
        Mask = Flux.param(LinearAlgebra.tril!(trues(size(Mask)), -1) * (-1e9))
        Numerator = Numerator .* Mask
    end

    x = TensorSoftmax(Numerator / Denominator, dims=2)
    x = [@inbounds x[:, :, batch] * Value[:, :, batch] for batch in 1:size(Query, 3)]
    return cat(x..., dims=3)
end

function _MHALayer(MHA::MultiHeadAttention, MHAData::AbstractArray{T, 1} where T, L::Integer; mask = false)
    (Query, Key, Value) = MHAData
    Nbatches = size(Query, 3)

    HQueries = cat([@inbounds Query[:, :, batch] * MHA.WQueries[L] for batch in 1:Nbatches]..., dims=3)
    HKeys = cat([@inbounds Key[:, :, batch] * MHA.WKeys[L] for batch in 1:Nbatches]..., dims=3)
    HValues = cat([@inbounds Value[:, :, batch] * MHA.WValues[L] for batch in 1:Nbatches]..., dims=3)

    return ScalarDotAttention(HQueries, HKeys, HValues)
end

function (m::MultiHeadAttention)(Query::AbstractArray{T, 3} where T, Key::AbstractArray{T, 3} where T, Value::AbstractArray{T, 3} where T; layers = 1, mask = false)
    TotalHeads = cat([_MHALayer(m, [Query, Key, Value], layer) for layer in 1:m.L]..., dims=2)
    return cat([@inbounds TotalHeads[:, :, batch] * m.HeadDense[:, :] for batch in 1:size(Query, 3)]..., dims=3)
end
