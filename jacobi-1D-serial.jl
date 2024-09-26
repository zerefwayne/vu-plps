function jacobi(n, n_iters)
    u = zeros(n + 2)
    u[1] = -1
    u[end] = 1

    u_new = copy(u)

    for t in 1:n_iters
        for i in 2:(n+1)
            u_new[i] = 0.5 * (u[i-1] + u[i+1])
        end
        u, u_new = u_new, u
    end
    u
end

function jacobi_tol(n, tol)
    u = zeros(n + 2)
    u[1] = -1
    u[end] = 1

    u_new = copy(u)

    while true
        diff = 0.0
        for i in 2:(n+1)
            u_new[i] = 0.5 * (u[i-1] + u[i+1])
            diff = max(abs(u_new[i] - u[i]), diff)
        end

        if diff < tol
            return u_new
        end

        u, u_new = u_new, u
    end
    u
end

N = 10

@show jacobi(N, 0)
@show jacobi(N, 10)
@show jacobi(N, 10000)

println()

@show jacobi_tol(N, 1)
@show jacobi_tol(N, 0.1)
@show jacobi_tol(N, 1e-10)
