using Flux: TrackedArray

SequenceLengths(x::AbstractArray{T, 1} where T) = [Int32(size(x[batch], 1)) for batch in 1:size(x, 1)]

function SequenceMask(x::AbstractArray{T, 3} where T, seqLens::AbstractArray{T, 1} where T)
    (T, D, N) = size(x)
    mask = zeros(Float32, T, D, N)
    for batch in 1:N
        len = seqLens[batch]
        mask_ones = ones(Float32, len, D)
        mask_zeros = zeros(Float32, T - len, D)
        mask_batch = reduce(vcat, [mask_ones, mask_zeros])
        mask[:, :, batch] = mask_batch
    end
    return mask
end

function _PadMerge(x::TrackedArray, pad::AbstractArray)
    xs = reduce(vcat, [x, pad])
    return Flux.Tracker.collect(xs)
end

function SequencePad(x::AbstractArray{T, 1} where T, maxLen::Integer)
    seqLens = SequenceLengths(x)
    xs = [
        _PadMerge(x[batch], param(zeros(Float32, maxLen - seqLens[batch], size(x[batch], 2))))
        for batch in 1:size(x, 1)
    ]
    return cat(xs..., dims=3)
end
