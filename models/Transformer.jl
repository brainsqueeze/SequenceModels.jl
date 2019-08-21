import Flux

include("../src/TensorOps.jl")
include("../src/Embedding.jl")
include("../src/PositionalEncoder.jl")
include("../src/Attention.jl")
include("../src/SequenceTools.jl")
include("../src/MultiHead.jl")
include("../src/FeedForward.jl")

struct SequenceFeed
    T::Integer
    V::Integer
    Emb
end

SequenceFeed(Dims::Integer, VocabSize::Integer, TimeSteps::Integer) = SequenceFeed(TimeSteps, VocabSize, Embedding(VocabSize, Dims))
Flux.@treelike SequenceFeed

(m::SequenceFeed)(x::AbstractArray{T, 1} where T) = SequencePad(m.Emb(x), m.T)

struct Encoding
    S::Integer
    IDrop
    HDrop
    PosEnc
    MHA
    FFN
end

Encoding(Dims::Integer, TimeSteps::Integer, MHALayers::Integer, MHAStacks::Integer) = Encoding(
    MHAStacks,
    Dropout(0.1), Dropout(0.1),
    PositionEncoder(TimeSteps, Dims), MultiHeadAttention(Dims, MHALayers),
    PointwiseFeedForward(Dims))
Flux.@treelike Encoding

function (m::Encoding)(x::AbstractArray{T, 3}, SeqLens::AbstractArray{Int32, 1}) where T
    Mask = SequenceMask(x, SeqLens)

    x = m.PosEnc(x, Mask)
    x = m.IDrop(x)

    # multi-head attention
    for _ in 1:m.S
        x = m.MHA(x, x, x)
    end
    x = m.HDrop(x) .+ x
    x = BatchLayerNorm(x)

    # feed forward network
    x = m.FFN(x)
    x = m.HDrop(x) .+ x
    x = BatchLayerNorm(x)
    return x, Mask
end

struct Decoding
    S::Integer
    IDrop
    HDrop
    PosEnc
    MHA
    FFN
end

Decoding(Dims::Integer, TimeSteps::Integer, MHALayers::Integer, MHAStacks::Integer) = Decoding(
    MHAStacks,
    Dropout(0.1), Dropout(0.1),
    PositionEncoder(TimeSteps, Dims), MultiHeadAttention(Dims, MHALayers),
    PointwiseFeedForward(Dims))
Flux.@treelike Decoding

function (m::Decoding)(x::AbstractArray{T, 3}, y::AbstractArray{T, 3}, SeqLens::AbstractArray{Int32, 1}, Context::AbstractArray{T, 2}, EncMask::AbstractArray{T, 3}, Attn::Attention) where T
    Mask = SequenceMask(x, SeqLens)

    y = m.PosEnc(y, Mask)
    y = m.IDrop(y)

    # multi-head attention
    for _ in 1:m.S
        y = m.MHA(y, y, y, futuremask=true)
    end
    y = m.HDrop(y) .+ y
    y = BatchLayerNorm(y)

    CrossContext = Attn(x .* EncMask, y .* Mask)
    y = Projection(y, CrossContext)
    y = m.HDrop(y) .+ y
    y = BatchLayerNorm(y)
    y = m.FFN(y)
    y = m.HDrop(y) .+ y
    y = BatchLayerNorm(y)

    y = Projection(y, Context)
    return m.HDrop(y) .+ y
end


struct DenseProjection{T}
    Bias::AbstractArray{T, 2}
end

DenseProjection(VocabSize::Integer) = DenseProjection(Flux.param(Flux.gpu(zeros(Float32, 1, VocabSize))))
Flux.@treelike DenseProjection

function (m::DenseProjection)(x::AbstractArray{T, 3}, W::AbstractArray{T, 2}) where T
    x_out = TensorDot(x, permutedims(W, [2, 1]))
    return x_out .+ m.Bias
end
