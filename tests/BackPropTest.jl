import CuArrays

include("../models/Transformer.jl")
include("../src/TensorOps.jl")

const TimeSteps = 100
const EmbDims = 128
const Vocab = Integer(1e3)
const Layers = 8
const Stacks = 1

const X = map(a->Int32.(a), [[3,7,3,5], [8,6,4,5,2,1], [10,7,5], [11,3]]);
const Y = map(a->Int32.(a), [[3,7,3,5], [8,6,4,5,2,1], [10,7,5], [11,3]]);

const Input = SequenceFeed(EmbDims, Vocab, TimeSteps)
const Encode = Encoding(EmbDims, TimeSteps, Layers, Stacks)
const Decode = Decoding(EmbDims, TimeSteps, Layers, Stacks)
const Output = DenseProjection(Vocab)
const Attn = Attention(EmbDims)
const θ = Flux.params(Input, Encode, Decode, Output, Attn)

const opt = Flux.ADAM(0.1);

function Model(x::AbstractArray{T, 1}, y::AbstractArray{T, 1}) where T
    EncSeqLens = SequenceLengths(x)
    x = Input(x)
    x, EncMask = Encode(x, EncSeqLens)
    context = Attn(x .* EncMask)

    DecSeqLens = SequenceLengths(y)
    y = Input(y)
    y = Decode(x, y, DecSeqLens, context, EncMask, Attn)
    y = Output(y, Input.Emb.E)
    return y
end

function Loss(x::AbstractArray{T, 1}, y::AbstractArray{T, 1}) where T
    ŷ = Model(x, y)
    # GC.gc()
    (Steps, Labels, Batches) = size(ŷ)

    y = map(seq -> Flux.param(Float32.(permutedims(Flux.onehotbatch(seq, 1:Labels), [2, 1]))), y)
    y = ArrayPad(y, Steps)
    s = TensorSoftmax(ŷ, dims=2)
    cost = y .* log.(s) + (1 .- y) .* log.(1 .- s)
    return - sum(cost) / prod(size(cost))
end

# @time loss = Loss(X, Y);
# println(loss)
# @time grads = Flux.Tracker.gradient(() -> loss, θ);
# @time Flux.Tracker.update!(opt, θ, grads)

# println(grads.grads)

@time Flux.train!(Loss, θ, collect(zip(X, Y)), opt)
