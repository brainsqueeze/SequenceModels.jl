using CuArrays

include("../models/Transformer.jl")

TimeSteps = 100
EmbDims = 128
Vocab = Integer(1e4)
Layers = 8
Stacks = 1
Model = SequenceModel(EmbDims, Vocab, TimeSteps, Layers, Stacks)

X = map(a->Int32.(a), [[3,7,3,5], [8,6,4,5,2,1], [10,7,5], [11,3]]);
Y = map(a->Int32.(a), [[3,7,3,5], [8,6,4,5,2,1], [10,7,5], [11,3]]);
@time (output, context, decoutput, crosscontext) = Model(X, Y);
println(size(output))

# @time Model(X, Y);

println(crosscontext)
