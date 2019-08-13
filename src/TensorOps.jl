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

    Inner = gpu(cat([@inbounds x[:, :, batch] * P[:, batch] for batch in 1:size(x, 3)]..., dims=2))
    NormSqr = sum(P .^ 2, dims=1)
    NormSqr = repeat(NormSqr, outer=[TimeSteps, 1])
    
    α = Inner ./ NormSqr
    return cat([@inbounds α[:, batch] .* P[:, batch]' for batch in 1:size(α, 2)]..., dims=3)
end
