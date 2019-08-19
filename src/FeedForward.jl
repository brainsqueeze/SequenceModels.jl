import Flux

struct PointwiseFeedForward
    FirstFilter::Flux.Conv
    SecondFilter::Flux.Conv
end

PointwiseFeedForward(dims::Integer) = PointwiseFeedForward(
    Flux.Conv((1, 1), dims => 4 * dims, Flux.relu) |> Flux.gpu,
    Flux.Conv((1, 1), 4 * dims => dims) |> Flux.gpu
)
Flux.@treelike PointwiseFeedForward

function (m::PointwiseFeedForward)(x::AbstractArray{T, 3} where T) 
    # x = permutedims(repeat(x, outer=[1, 1, 1, 1]), [1, 4, 2, 3])
    (T, D, N) = size(x)
    x = reshape(x, (T, 1, D, N))
    x = m.SecondFilter(m.FirstFilter(x))
    return dropdims(x, dims=2)
end