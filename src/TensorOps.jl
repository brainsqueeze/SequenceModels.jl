function TensorSoftmax(x::AbstractArray{T, 3} where T; dims = 3)
    Soft = exp.(x)
    return Soft ./ sum(Soft, dims=dims)
end

mean(x::AbstractArray, dims::Integer) = sum(x, dims=dims) / size(x, dims)

function BatchLayerNorm(x::AbstractArray{T, 3} where T; ϵ = 1e-8, scale = 1.0, bias = 0)
    average = mean(x, 3)
    variance = mean((x .- average) .^ 2, 3)
    xnorm = (x .- average) ./ ((variance .+ ϵ) .^ 0.5)
    return xnorm .* scale .+ bias
end

