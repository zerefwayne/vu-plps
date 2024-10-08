# Custom types
const DistanceMatrix = Matrix{Float64}
const SortedCitiesDistances = Vector{Vector{Tuple{Int64,Float64}}}
const Path = Vector{Int64}

function sort_neighbours(C::DistanceMatrix)::SortedCitiesDistances
    n::Int64 = size(C, 1)
    map(1:n) do i
        Ci = C[i, :]
        cities = sortperm(Ci)
        distances = Ci[cities]
        collect(zip(cities, distances))
    end
end

function traverse_from(start::Int64, C_sorted::SortedCitiesDistances)::Nothing
    n::Int64 = length(C_sorted)
    hops::Int64 = 1
    path::Path = zeros(Int64, n)
    path[hops] = start
    traverse_all_from_recursive!(hops, n, path, C_sorted)
end


function traverse_all_from_recursive!(hops::Int64, n::Int64, path::Path, C_sorted::SortedCitiesDistances)::Nothing
    if hops != n
        current = path[hops]
        connections = C_sorted[current]
        for (next_city, distance_increment) in connections
            already_visited = (next_city in view(path, 1:hops))
            if !already_visited
                path[hops+1] = next_city
                traverse_all_from_recursive!(hops + 1, n, path, C_sorted)
            end
        end
    else
        println(path)
    end
    return nothing
end

function traverse_all_no_prune(start::Int64, C_sorted::SortedCitiesDistances)::Float64
    n::Int64 = length(C_sorted)
    hops::Int64 = 1
    path::Path = zeros(Int64, n)
    path[hops] = start
    distance::Float64 = 0.0
    min_distance::Float64 = typemax(Float64)
    traverse_all_no_prune_recursive!(hops, n, distance, min_distance, path, C_sorted)
end

function traverse_all_no_prune_recursive!(hops::Int64, n::Int64, distance::Float64, min_distance::Float64, path::Path, C_sorted::SortedCitiesDistances)::Float64
    if hops != n
        current = path[hops]
        connections = C_sorted[current]
        for (next_city, distance_increment) in connections
            already_visited = (next_city in view(path, 1:hops))
            if !already_visited
                next_distance = distance + distance_increment
                path[hops+1] = next_city
                min_distance = min(traverse_all_no_prune_recursive!(hops + 1, n, next_distance, min_distance, path, C_sorted), min_distance)
            end
        end
    else
        min_distance = min(min_distance, distance)
    end
    return min_distance
end

function traverse_all_prune(start::Int64, C_sorted::SortedCitiesDistances)::Float64
    n::Int64 = length(C_sorted)
    hops::Int64 = 1
    path::Path = zeros(Int64, n)
    path[hops] = start
    distance::Float64 = 0.0
    min_distance::Float64 = typemax(Float64)
    traverse_all_prune_recursive!(hops, n, distance, min_distance, path, C_sorted)
end

function traverse_all_prune_recursive!(hops::Int64, n::Int64, distance::Float64, min_distance::Float64, path::Path, C_sorted::SortedCitiesDistances)::Float64
    if distance >= min_distance
        return min_distance
    end
    if hops != n
        current = path[hops]
        connections = C_sorted[current]
        for (next_city, distance_increment) in connections
            already_visited = (next_city in view(path, 1:hops))
            if !already_visited
                next_distance = distance + distance_increment
                path[hops+1] = next_city
                min_distance = min(traverse_all_prune_recursive!(hops + 1, n, next_distance, min_distance, path, C_sorted), min_distance)
            end
        end
    else
        min_distance = min(min_distance, distance)
    end
    return min_distance
end

N = 12

C::DistanceMatrix = rand(Float64, N, N) .* 100
for i in 1:N
    C[i, i] = 0
end

C_sorted::SortedCitiesDistances = sort_neighbours(C)

min_distance_from_1_no_prune = @time traverse_all_no_prune(1, C_sorted)

println("Minimum distance without prune: ", min_distance_from_1_no_prune)

min_distance_from_1_prune = @time traverse_all_prune(1, C_sorted)

println("Minimum distance with prune: ", min_distance_from_1_prune)

@assert min_distance_from_1_no_prune ≈ min_distance_from_1_prune
