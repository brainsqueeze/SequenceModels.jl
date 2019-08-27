include("./DataTools.jl")
include("./Transformer.jl")

const RootDir = "./models/AttentionEmbed/"

sentences = vcat(TextToArray(RootDir * "data/plots.txt"), TextToArray(RootDir * "data/quotes.txt"))
cvData, trainData = DatasetSplit(sentences, cvPercent = 0.2)

const NumEpochs = 5
const BatchSize = 32
const NumBatches = 20

const Vocab = Int32(1e3)
const ModelVocab = Vocab + 3
const EmbDims = 128
const TimeSteps = GetMaxSequenceLength(trainData)
const Layers = 8
const Stacks = 1

const Input = SequenceFeed(EmbDims, ModelVocab, TimeSteps)
const Encode = Encoding(EmbDims, TimeSteps, Layers, Stacks)
const Decode = Decoding(EmbDims, TimeSteps, Layers, Stacks)
const Output = DenseProjection(ModelVocab)

const θ = Flux.params(Input, Encode, Decode, Output)
const opt = Flux.ADAM(0.001)

const Lookup = TopNLookup(TokenFrequencies(sentences), Vocab)
const cvX = TokenizeLookup(cvData, Lookup)

function MakeDecodeTarget(x::AbstractArray{String, 1})
    x = Tokenize(x)
    target = TagEnding(x)
    decode = TagBeginning(x)

    target = TokenizeLookup(map(s -> join(s, " "), target), Lookup)
    decode = TokenizeLookup(map(s -> join(s, " "), decode), Lookup)
    return decode, target
end

function Model(x::AbstractArray{T, 1}, y::AbstractArray{T, 1}) where T
    EncSeqLens = SequenceLengths(x)
    x = Input(x)
    x, EncMask, context = Encode(x, EncSeqLens)

    DecSeqLens = SequenceLengths(y)
    y = Input(y)
    y = Decode(x, y, DecSeqLens, context, EncMask, Encode.Attn)
    y = Output(y, Input.Emb.E)
    return y
end

function Loss(x::AbstractArray{T, 1}, xdecode::AbstractArray{T, 1}, y::AbstractArray{T, 1}) where T
    ϵ = Float32(1e-8)
    ŷ = Model(x, xdecode)
    (Steps, Labels, Batches) = size(ŷ)

    y = map(seq -> Flux.data(Float32.(permutedims(Flux.onehotbatch(seq, 1:Labels), [2, 1]))), y)
    y = SequencePad(y, Steps)
    s = TensorSoftmax(ŷ, dims=2)
    cost = y .* log.(s .+ ϵ) + (1 .- y) .* log.(1 .- s .+ ϵ)
    return - sum(cost) / prod(size(cost))
end

for epoch in 1:NumEpochs
    println("[INFO] Training on epoch $epoch")
    _sent = Random.shuffle(trainData)[1:(BatchSize * NumBatches)]

    for minibatch in 1:NumBatches
        start = minibatch * BatchSize
        stop = (minibatch + 1) * BatchSize

        if start > length(_sent) || stop > length(_sent)
            break
        end

        sents = _sent[start:stop]
        X = TokenizeLookup(sents, Lookup)
        XDecode, Y = MakeDecodeTarget(sents)
        loss = Loss(X, XDecode, Y)
        println("\t mini-batch loss: $loss")

        if isnan(loss)
            Ŷ = Model(X, XDecode)
            s = TensorSoftmax(Ŷ, dims=2)
            println("model output check: $(size(Ŷ[isnan.(Ŷ)]))")
            println("nan check: $(size(Ŷ[isnan.(s)]))")
        end

        grads = Flux.Tracker.gradient(() -> loss, θ)
        Flux.Tracker.update!(opt, θ, grads)
    end
end
