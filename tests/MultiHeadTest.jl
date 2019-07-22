include("PosEncodeTest.jl")
include("../src/MultiHead.jl")

Layers = 8
@assert Int(EmbDims / Layers) - (EmbDims / Layers) == 0

MHA = MultiHeadAttention(EmbDims, Layers)
@time output = MHA(output)