using PrettyTables

macro pprint(matrix)
    quote
        println("Matrix: ", $(string(matrix)))
        pretty_table($matrix)
        println()
    end
end

N::Int64 = 10

function gaussian_elimination!(N::Int64, augmented_matrix::Matrix{Float64}) 
    # Iterate over all N columns except the last one
    for col::Int64 in 1:N
        # Check if the leading entry is zero and swap with a non-zero row
        if augmented_matrix[col, col] == 0
            for row::Int64 in (col+1):N
                if augmented_matrix[row, col] != 0
                    augmented_matrix[col, :], augmented_matrix[row, :] = augmented_matrix[row, :], augmented_matrix[col, :]
                    break
                end
            end
        end

        # Set leading entry of row `col` to 1
        augmented_matrix[col, :] /= augmented_matrix[col, col]

        # Set all following rows to 0
        for row::Int64 in (col+1):N
            factor = augmented_matrix[row, col] / augmented_matrix[col, col]
            augmented_matrix[row, :] .= augmented_matrix[row, :] .- factor .* augmented_matrix[col, :]
        end
    end
end

# N * (N+1), 1 extra column for RHS
augmented_matrix = rand(Float64, N, N+1)

@pprint augmented_matrix

println("Applying gaussian elimination ⏳\n")

gaussian_elimination!(N, augmented_matrix)

@pprint augmented_matrix
