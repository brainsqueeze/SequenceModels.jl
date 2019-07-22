include("../src/Embedding.jl")

TimeSteps = 100
EmbDims = 128
Vocab = Integer(1e4)

Embed = Embedding(Vocab, EmbDims)
X = map(a->Int32.(a), [[3,7,3,5], [8,6,4,5,2,1], [10,7,5], [11,3]])
@time output = Embed(X)
