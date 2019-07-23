using Flux: param, TrackedArray, gpu

function build_encoder(seqLen::Integer, dims::Integer)
    positions = 1:seqLen
    columns = 1:dims
    factor = (1e5 ^ (2 / dims)) .^ columns
    factor = reshape(factor, size(factor, 1), 1)

    even = sin.(positions / factor[1:2:end, :])
    odd = cos.(positions / factor[2:2:end, :])

    encoder = zeros(Float32, seqLen, dims)
    encoder[:, 1:2:end] = even
    encoder[:, 2:2:end] = odd
    return encoder
end

struct Encoder
    E::TrackedArray{Float32, 2}
end

Encoder(seqLen::Integer, dims::Integer) = Encoder(param(build_encoder(seqLen, dims)))
function (m::Encoder)(x::AbstractArray{T, 3} where T, mask::AbstractArray{T, 3} where T)
    maxLen = size(m.E, 1)
    x = x[1:min(size(x, 1), maxLen), :, :]
    return m.E .* mask
end
