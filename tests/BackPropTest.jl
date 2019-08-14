using CuArrays

include("../models/Transformer.jl")
include("../src/TensorOps.jl")

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

function Loss(y::AbstractArray{T, 1} where T, ŷ::AbstractArray{T, 3} where T)
    (Steps, Labels, Batches) = size(ŷ)

    y = map(seq -> Flux.param(Float32.(transpose(Flux.onehotbatch(seq, 1:Labels)))), y)
    y = ArrayPad(y, Steps)
    s = TensorSoftmax(ŷ, dims=2)

    cost = y .* log.(s) + (1 .- y) .* log.(1 .- s)
    return Flux.Tracker.data(- sum(cost) / prod(size(cost)))
end

@time loss = Loss(Y, Model(X, Y));
println(loss)

@time grads = Flux.Tracker.gradient(() -> loss, Flux.params(Input, Encode, Decode, Output, Attn))
println(grads)