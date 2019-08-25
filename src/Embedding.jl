using Flux

struct Embedding{T}
    E::Flux.TrackedArray{T, 2}
end

Embedding(vocab::Integer, dims::Integer) = Embedding(Flux.param(Flux.gpu(randn(Float32, vocab, dims))))
SequenceTransform(x::AbstractArray{T, 1} where T, vsize::Integer) = permutedims(Float32.(Flux.onehotbatch(x, 1:vsize)), [2, 1]) |> Flux.gpu
(m::Embedding)(x::AbstractArray{T, 1} where T) = map(seq -> SequenceTransform(seq, size(m.E, 1)) * m.E, x)
Flux.@treelike Embedding
