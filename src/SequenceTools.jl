using Flux: gpu
using Flux.Tracker: data, collect

SequenceLengths(x::AbstractArray{T, 1} where T) = [Int32(size(x[batch], 1)) for batch in 1:size(x, 1)]

function SequenceMask(x::AbstractArray{T, 3} where T, seqLens::AbstractArray{T, 1} where T)
    (S, D, N) = size(x)
    mask = zeros(Float32, S, D, N)
    for batch in 1:N
        len = seqLens[batch]
        @inbounds mask[:, :, batch] = vcat(ones(Float32, len, D), zeros(Float32, S - len, D))
    end
    return data(gpu(mask))
end

_MakePad(rows::Integer, columns::Integer) = data(gpu(zeros(Float32, rows, columns)))
_PadMerge(x::TrackedArray, pad::AbstractArray) = collect(vcat(x, pad))

function SequencePad(x::AbstractArray{T, 1} where T, maxLen::Integer)
    seqLens = SequenceLengths(x)
    return gpu(cat([_PadMerge(x[batch], _MakePad(maxLen - seqLens[batch], size(x[batch], 2))) for batch in 1:size(x, 1)]..., dims=3))
end
