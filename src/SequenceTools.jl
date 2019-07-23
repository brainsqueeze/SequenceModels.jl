using Flux

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
    return Flux.param(mask) |> gpu
end

_PadMerge(x::TrackedArray, pad::AbstractArray) = Flux.Tracker.collect(vcat(x, pad))

function SequencePad(x::AbstractArray{T, 1} where T, maxLen::Integer)
    seqLens = SequenceLengths(x)
    return cat([_PadMerge(x[batch], param(Flux.gpu(zeros(Float32, maxLen - seqLens[batch], size(x[batch], 2))))) for batch in 1:size(x, 1)]..., dims=3)
end
