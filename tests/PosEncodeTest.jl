include("PadTest.jl")
include("../src/PositionalEncoder.jl")

Enc = Encoder(TimeSteps, EmbDims)

SeqLens = SequenceLengths(X)
mask = Flux.param(SequenceMask(output, SeqLens))
@time output = Enc(output, mask) .+ output
