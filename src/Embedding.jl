using Flux

struct Embedding{T}
    E::Flux.TrackedArray{T, 2}
end

Embedding(vocab::Integer, dims::Integer) = Embedding(Flux.param(randn(Float32, vocab, dims)))
(m::Embedding)(x::AbstractArray{T, 1} where T) = map(seq -> transpose(Flux.onehotbatch(seq, 1:size(m.E, 1))) * m.E, x)
Flux.@treelike Embedding
