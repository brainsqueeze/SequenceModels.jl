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
    TimeSteps, 
    VocabSize, 
    Dims,
    MHAStacks,
    Dropout(0.1),
    Dropout(0.1),
    Embedding(VocabSize, Dims), 
    Encoder(TimeSteps, Dims),
    Attention(Dims),
    MultiHeadAttention(Dims, MHALayers),
    PointwiseFeedForward(Dims))
Flux.@treelike SequenceModel

function (m::SequenceModel)(x::AbstractArray{T, 1} where T)
    SeqLens = SequenceLengths(x)

    x = m.Emb(x)
    x = SequencePad(x, m.T)
    mask = Flux.param(SequenceMask(x, SeqLens))
    x = m.Enc(x, mask) .+ x
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
    context = m.Attn(x .* mask)
    return x, context
end
