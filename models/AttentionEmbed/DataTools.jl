import Random

TextToArray(filename::String) = open(filename) do file
    texts = [line for line in eachline(file)]
end

TagBeginning(x::AbstractArray; tag = "<s>") = map(tokens -> vcat([tag], tokens), x)
TagEnding(x::AbstractArray; tag = "</s>") = map(tokens -> vcat(tokens, [tag]), x)

function TokenFrequencies(x::AbstractArray{String, 1}, sep = r"\s+")
    Lookup = Dict{String, Int32}()
    for i in 1:size(x, 1)
        for token in split(x[i], sep)
            if haskey(Lookup, token)
                Lookup[token] += Int32(1)
            else
                Lookup[token] = Int32(1)
            end
        end
    end
    return Lookup
end

GetMaxSequenceLength(x::AbstractArray{String, 1}, sep = r"\s+") = mapreduce(s -> length(split(s, sep)) |> Int32, max, x)
GetTopNTokens(TokenCounts::Dict{String, Int32}, N::Integer) = sort([[kv...] for kv in TokenCounts], lt=(x, y) -> isless(x[2], y[2]), rev = true)[1:N]

function TopNLookup(TokenCounts::Dict{String, Int32}, N::Integer)
    Lookup = Dict{String, Int32}([
        [item[2][1], item[1]]
        for item in enumerate(GetTopNTokens(TokenCounts, N))
    ])
    MaxVal = values(Lookup) |> maximum
    Lookup["<unk>"] = MaxVal + 1
    Lookup["<s>"] = MaxVal + 2
    Lookup["</s>"] = MaxVal + 3
    return Lookup
end

Tokenize(x::AbstractArray{String, 1}, sep = r"\s+") = map(s -> split(s, sep), x)
function TokensHash(s::String, TokenLookup::Dict{String, Int32}, sep = r"\s+")
    stringTokens = split(s, sep)
    out = similar(stringTokens, Int32)

    for item in enumerate(stringTokens)
        pos, token = item
        out[pos] = haskey(TokenLookup, token) ? TokenLookup[token] : TokenLookup["<unk>"]
    end
    return out
end
TokenizeLookup(x::AbstractArray{String, 1}, TokenLookup::Dict{String, Int32}, sep = r"\s+") = map(s -> TokensHash(s, TokenLookup, sep), x)
function DatasetSplit(x::AbstractArray{String, 1}; cvPercent = 0.1)
    if cvPercent >= 0.0 && cvPercent < 1.0
        x = Random.shuffle(x)
        cvSize = Int32(100 * cvPercent)
        return x[1:cvSize], x[cvSize:end]
    end
    return [], x
    
        
end
