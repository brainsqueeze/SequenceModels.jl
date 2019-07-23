using Flux

include("../src/TensorOps.jl")
include("../src/Embedding.jl")
include("../src/PositionalEncoder.jl")
include("../src/Attention.jl")
include("../src/SequenceTools.jl")
include("../src/MultiHead.jl")
include("../src/FeedForward.jl")

struct SequenceModel
    T::Integer
    V::Integer
    D::Integer
    S::Integer
    IDrop
    HDrop
    Emb
    Enc
    Attn
    MHA
    FFN
end

SequenceModel(Dims::Integer, VocabSize::Integer, TimeSteps::Integer, MHALayers::Integer, MHAStacks::Integer) = SequenceModel(
    TimeSteps, VocabSize, Dims, MHAStacks,
    Dropout(0.1), Dropout(0.1),
    Embedding(VocabSize, Dims), Encoder(TimeSteps, Dims),
    Attention(Dims), MultiHeadAttention(Dims, MHALayers),
    PointwiseFeedForward(Dims))
Flux.@treelike SequenceModel

function (m::SequenceModel)(x::AbstractArray{T, 1} where T, y::AbstractArray{T, 1} where T)
    EncSeqLens = SequenceLengths(x)
    DecSeqLens = SequenceLengths(y)

    x = m.Emb(x)
    x = SequencePad(x, m.T)
    EncMask = Flux.param(SequenceMask(x, EncSeqLens))
    x = m.Enc(x, EncMask) .+ x
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

    # Bahdanau attention
    context = m.Attn(x .* EncMask)

    y = m.Emb(y)
    y = SequencePad(y, m.T)
    DecMask = Flux.param(SequenceMask(y, DecSeqLens))
    y = m.Enc(y, DecMask) .+ y
    y = m.IDrop(y)

    # multi-head attention
    for _ in 1:m.S
        y = m.MHA(y, y, y, futuremask=true)
    end
    y = m.HDrop(y) .+ y
    println(y[:, :, 1])
    y = BatchLayerNorm(y)

    CrossContext = m.Attn(x .* EncMask, y .* DecMask)
    return x, context, y, CrossContext
end
