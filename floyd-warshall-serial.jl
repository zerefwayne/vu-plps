macro pprint(matrix)
    quote
        using PrettyTables

        println("Matrix: ", $(string(matrix)))
        pretty_table($matrix)
        println()
    end
end

function floyd_warshall!(dp::Matrix{Float64}, next::Matrix{Int64}; negative_cycle_mode::Bool=false)
    negative_cycle_detected = false

    for k in 1:N
        for i in 1:N, j in 1:N
            distance_via_k = dp[i, k] + dp[k, j]

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

    if negative_cycle_mode
        if negative_cycle_detected
            println("Detected a negative cycle!\n")
        else
            println("No negative cycles detected!\n")
        end
    end
end

function reconstruct_path(next::Matrix{Int64}, src::Int64, dest::Int64)
    path = [src]
    while src != dest
        src = next[src, dest]
        src == -1 && return nothing
        push!(path, src)
    end
    path
end

function reconstruct_all_paths(next::Matrix{Int64})
    for src in 1:N
        for dest in 1:N
            if src != dest
                path = reconstruct_path(next, src, dest)
                println("$src -> $dest: ", path !== nothing ? path : "nope")
            end
        end
    end
end

N::Int64 = 4

m = [
    0.0 Inf Inf 1.0;
    2.0 0.0 3.0 9.0;
    Inf Inf 0.0 Inf;
    Inf Inf 5.0 0.0
]::Matrix{Float64}

dp = zeros(Float64, N, N)
next = zeros(Int64, N, N)

dp .= m
next .= ifelse.(m .!= Inf, collect(1:N)', -1)

floyd_warshall!(dp, next)
floyd_warshall!(dp, next; negative_cycle_mode=true)

@pprint m

reconstruct_all_paths(next)
