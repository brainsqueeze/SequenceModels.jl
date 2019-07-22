using Flux

include("../src/Embedding.jl")
include("../src/PositionalEncoder.jl")
include("../src/Attention.jl")
include("../src/SequenceTools.jl")
include("../src/MultiHead.jl")

struct SequenceModel
    T::Integer
    V::Integer
    D::Integer
    S::Integer
    Emb
    Enc
    Attn
    MHA
end

SequenceModel(Dims::Integer, VocabSize::Integer, TimeSteps::Integer, MHALayers::Integer, MHAStacks::Integer) = SequenceModel(
    TimeSteps, 
    VocabSize, 
    Dims,
    MHAStacks,
    Embedding(VocabSize, Dims), 
    Encoder(TimeSteps, Dims),
    Attention(Dims),
    MultiHeadAttention(Dims, MHALayers))
Flux.@treelike SequenceModel

function (m::SequenceModel)(x::AbstractArray{T, 1} where T)
    SeqLens = SequenceLengths(x)

    x = m.Emb(x)
    x = SequencePad(x, m.T)
    mask = Flux.param(SequenceMask(x, SeqLens))
    x = m.Enc(x, mask) .+ x
    
    for _ in 1:m.S
        x = m.MHA(x, x, x)
    end
    context = m.Attn(x)
    return x, context
end
