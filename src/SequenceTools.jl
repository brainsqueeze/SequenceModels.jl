using Flux

SequenceLengths(x::AbstractArray{T, 1} where T) = [Int32(size(x[batch], 1)) for batch in 1:size(x, 1)]

function SequenceMask(x::AbstractArray{T, 3} where T, seqLens::AbstractArray{T, 1} where T)
    (S, D, N) = size(x)
    mask = zeros(Float32, S, D, N)
    for batch in 1:N
        len = seqLens[batch]
        @inbounds mask[:, :, batch] = vcat(ones(Float32, len, D), zeros(Float32, S - len, D))
    end
    return Flux.param(mask) |> gpu
end

_MakePad(rows::Integer, columns::Integer) = Flux.param(Flux.gpu(zeros(Float32, rows, columns)))
_PadMerge(x::TrackedArray, pad::AbstractArray) = Flux.Tracker.collect(vcat(x, pad))

function SequencePad(x::AbstractArray{T, 1} where T, maxLen::Integer)
    seqLens = SequenceLengths(x)
    return cat([_PadMerge(x[batch], _MakePad(maxLen - seqLens[batch], size(x[batch], 2))) for batch in 1:size(x, 1)]..., dims=3)
end
