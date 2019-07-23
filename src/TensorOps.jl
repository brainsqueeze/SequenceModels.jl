function TensorSoftmax(x::AbstractArray{T, 3} where T; dims = 3)
    ϵ = 1e-8
    Soft = exp.(x) .+ ϵ
    return Soft ./ sum(Soft, dims=dims)
end

mean(x::AbstractArray, dims::Integer) = sum(x, dims=dims) / size(x, dims)

function BatchLayerNorm(x::AbstractArray{T, 3} where T; ϵ = 1e-8, scale = 1.0, bias = 0)
    average = mean(x, 3)
    variance = mean((x .- average) .^ 2, 3)
    xnorm = (x .- average) ./ ((variance .+ ϵ) .^ 0.5)
    return xnorm .* scale .+ bias
end

function Projection(x::AbstractArray{T, 3} where T, P::AbstractArray{T, 2} where T)
    Inner = cat([@inbounds x[:, :, batch] * P[:, batch] for batch in 1:size(x, 3)]..., dims=2)
    NormSqr = sum(P .^ 2, dims=2)
    α = Inner / NormSqr
    return cat([@inbounds α[:, batch] .* P[:, batch]' for batch in 1:size(α, 2)]..., dims=3)
end
