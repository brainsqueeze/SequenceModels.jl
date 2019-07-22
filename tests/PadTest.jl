include("EmbeddingTest.jl")
include("../src/SequenceTools.jl")

@time output = SequencePad(output, TimeSteps)
