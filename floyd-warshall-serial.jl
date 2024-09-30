using PrettyTables

macro pprint(matrix)
    quote
        println("Matrix: ", $(string(matrix)))
        pretty_table($matrix)
        println()
    end
end

function floyd_warshall!(dp::Matrix{Float64}, next::Matrix{Int64}; negative_cycle_mode::Bool=false)::Bool
    negative_cycle_detected::Bool = false
    for k::Int64 in 1:N
        for i::Int64 in 1:N
            for j::Int64 in 1:N
                distance_via_k::Float64 = dp[i, k] + dp[k, j]

                if negative_cycle_mode
                    if dp[i, k] != Inf && dp[k, j] != Inf && distance_via_k < dp[i, j]
                        negative_cycle_detected = true
                        dp[i, j] = -Inf
                        next[i, j] = -1
                    end
                else
                    if dp[i, k] != Inf && dp[k, j] != Inf && distance_via_k < dp[i, j]
                        dp[i, j] = distance_via_k
                        next[i, j] = next[i, k]
                    end
                end
            end
        end
    end

    return negative_cycle_detected
end

function reconstruct_path(next::Matrix{Int64}, src::Int64, dest::Int64)::Union{Nothing,Vector{Int64}}
    path::Vector{Int64} = [src]
    while src != dest
        src = next[src, dest]
        src == -1 && return nothing
        push!(path, src)
    end
    return path
end

function reconstruct_all_paths(dp::Matrix{Float64}, next::Matrix{Int64})::Nothing
    println("All possible paths:\n")
    for src::Int64 in 1:N
        for dest::Int64 in 1:N
            if src != dest
                path = reconstruct_path(next, src, dest)
                println("$(src) -> $(dest): ", path !== nothing ? path : "nope", " [$(dp[src, dest])]")
            end
        end
    end
end

N::Int64 = 7

m::Matrix{Float64} = fill(Inf, N, N)
for i in 1:N
    m[i, i] = 0.0
end

m[1, 2] = 2
m[2, 4] = 4
m[4, 3] = -6
m[3, 2] = 1
m[1, 3] = 1
m[4, 5] = 1
m[5, 7] = 3
m[6, 7] = 1
m[5, 6] = 1
m[4, 6] = 1

@pprint m

dp::Matrix{Float64} = zeros(Float64, N, N)
next::Matrix{Int64} = zeros(Int64, N, N)

dp .= m
next .= ifelse.(m .!= Inf, collect(1:N)', -1)

floyd_warshall!(dp, next)

@pprint dp
@pprint next

negative_cycle_detected::Bool = floyd_warshall!(dp, next; negative_cycle_mode=true)

if negative_cycle_detected
    println("Detected a negative cycle!\n")
else
    println("No negative cycles detected!\n")
end

reconstruct_all_paths(dp, next)
