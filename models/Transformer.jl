using Flux

include("../src/Embedding.jl")
include("../src/PositionalEncoder.jl")
include("../src/Attention.jl")
include("../src/SequenceTools.jl")

struct SequenceModel
    T::Integer
    V::Integer
    D::Integer
    Emb
    Enc
    Attn
end

SequenceModel(D::Integer, V::Integer, T::Integer) = SequenceModel(T, V, D,
    Embedding(V, D), Encoder(T, D),
    Attention(EmbDims))
function (m::SequenceModel)(x::AbstractArray{T,1} where T)
    SeqLens = SequenceLengths(x)

    x = m.Emb(x)
    x = SequencePad(x, m.T)
    mask = Flux.param(SequenceMask(x, SeqLens))
    x = m.Enc(x, mask) .+ x
    x = m.Attn(x)
    return x
end

Flux.@treelike SequenceModel
