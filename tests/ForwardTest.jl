include("../models/Transformer.jl")

TimeSteps = 100
EmbDims = 128
Vocab = Integer(1e4)
Layers = 8
Stacks = 1

X = map(a->Int32.(a), [[3,7,3,5], [8,6,4,5,2,1], [10,7,5], [11,3]]);
Y = map(a->Int32.(a), [[3,7,3,5], [8,6,4,5,2,1], [10,7,5], [11,3]]);

Input = SequenceFeed(EmbDims, Vocab, TimeSteps)
Encode = Encoding(EmbDims, TimeSteps, Layers, Stacks)
Decode = Decoding(EmbDims, TimeSteps, Layers, Stacks)
Output = DenseProjection(Vocab)
Attn = Attention(EmbDims)

function Model(x::AbstractArray{T, 1}, y::AbstractArray{T, 1}) where T
    EncSeqLens = SequenceLengths(x)
    x = Input(x)
    x, EncMask = Encode(x, EncSeqLens)
    context = Attn(x .* EncMask)

    DecSeqLens = SequenceLengths(y)
    y = Input(y)
    y = Decode(x, y, DecSeqLens, context, EncMask, Attn)

    y = Output(y, Flux.gpu(Input.Emb.E))
    return y
end

@time output = Model(X, Y);
# println(output)
