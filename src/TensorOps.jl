using Flux: gpu

function TensorSoftmax(x::AbstractArray{T, 3} where T; dims = 3)
    max = maximum(x, dims=dims)
    return exp.(x .- max) ./ sum(exp.(x .- max), dims=dims)
end

mean(x::AbstractArray, dims::Integer) = sum(x, dims=dims) / size(x, dims)

function BatchLayerNorm(x::AbstractArray{T, 3} where T; ϵ = 1e-8, scale = 1.0, bias = 0)
    ϵ = Float32(ϵ)
    scale = Float32(scale)
    bias = Float32(bias)

    average = mean(x, 3)
    variance = mean((x .- average) .^ 2, 3)
    xnorm = (x .- average) ./ Float32.((variance .+ ϵ) .^ Float32(0.5))
    return xnorm .* scale .+ bias
end

function Projection(x::AbstractArray{T, 3}, P::AbstractArray{T, 2}) where T
    TimeSteps = size(x, 1)

    Inner = BatchAffineNoBias(x, P)
    NormSqr = sum(P .^ 2, dims=1)
    NormSqr = repeat(NormSqr, outer=[TimeSteps, 1])
    α = Inner ./ NormSqr

    if (typeof(α) <: Array)
        return cat([@inbounds α[:, batch] .* P[:, batch]' for batch in 1:size(α, 2)]..., dims=3)
    else
        return reshape(α, (size(α, 1), 1, size(α, 2))) .* reshape(P, (1, size(P, 1), size(P, 2)))
    end
end

function batchcuaffinenobias(x::AbstractArray{T, 3}, y::AbstractArray{T, 2}) where T
    y = reshape(y, (1, size(y, 1), size(y, 2)))
    return dropdims(sum(x .* y, dims=2), dims=2)
end
batchaffinenobias(x::AbstractArray{T, 3}, y::AbstractArray{T, 2}) where T = cat([@inbounds x[:, :, batch] * y[:, batch] for batch in 1:size(x, 3)]..., dims=2)
BatchAffineNoBias(x::AbstractArray{T, 3}, y::AbstractArray{T, 2}) where T = typeof(x.data) <: Array ? batchaffinenobias(x, y) : batchcuaffinenobias(x, y)

function cutensordot(x::AbstractArray{T, 3}, y::AbstractArray{T, 2}) where T
    x = reshape(x, (size(x, 1), size(x, 2), 1, size(x, 3)))
    y = reshape(y, (1, size(y, 1), size(y, 2), 1))
    return dropdims(sum(x .* y, dims=2), dims=2)
end
tensordot(x::AbstractArray{T, 3}, y::AbstractArray{T, 2}) where T = cat([@inbounds x[:, :, batch] * y for batch in 1:size(x, 3)]..., dims=3)
TensorDot(x::AbstractArray{T, 3}, y::AbstractArray{T, 2}) where T = typeof(x.data) <: Array ? tensordot(x, y) : cutensordot(x, y)

function batchcumatmul(x::AbstractArray{T, 3}, y::AbstractArray{T, 3}) where T
    x = reshape(x, (size(x, 1), size(x, 2), 1, size(x, 3)))
    y = reshape(y, (1, size(y, 1), size(y, 2), size(y, 3)))
    return dropdims(sum(x .* y, dims=2), dims=2)
end
batchmatmul(x::AbstractArray{T, 3}, y::AbstractArray{T, 3}) where T = cat([@inbounds x[:, :, batch] * y[:, :, batch] for batch in 1:size(x, 3)]..., dims=3)
BatchMatMul(x::AbstractArray{T, 3}, y::AbstractArray{T, 3}) where T = typeof(x.data) <: Array ? batchmatmul(x, y) : batchcumatmul(x, y)
