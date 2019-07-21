function TensorSoftmax(x::AbstractArray{T, 3} where T; dims = 3)
    Soft = exp.(x)
    return Soft ./ sum(Soft, dims=dims)
end
