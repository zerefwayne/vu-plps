# Implementation of MPI algorithm for Jacobi 1D with tolerance

using MPI

code = quote begin
        using MPI
        MPI.Init()

        comm = MPI.Comm_dup(MPI.COMM_WORLD)
        rank = MPI.Comm_rank(comm)
        root = 0
        
        P = MPI.Comm_size(comm)

        N, tolerance = 1000, 0.0002
        boundary_values = (-1.0, 2.0)

        # Cap tolerance to avoid infinite loop
        tolerance = max(abs(boundary_values[2] - boundary_values[1]) / N, tolerance)

        if mod(N, P) != 0
            println("ERROR: N ($N) is not a multiple of P ($P)\n")
            return
        end

        L = Int(N/P)

        # Step 1: Initialise array and ghost variables
        u = zeros(L)
        u_copy = zeros(L)
        ghost_prev, ghost_next = Ref(0.0), Ref(0.0)

        # Step 2: Set boundary values
        if rank == 0
            ghost_prev[] = boundary_values[1]
        end

        if rank == P - 1
            ghost_next[] = boundary_values[2]
        end

        function exchange_ghosts()
            # Send to next rank
            if rank < P - 1
                MPI.Send(u[end], comm; dest = rank + 1)
            end

            # Recieve from next rank
            if rank < P - 1
                receive_buffer = Ref(0.0)
                MPI.Recv!(receive_buffer, comm; source = rank + 1)
                ghost_next[] = receive_buffer[]
            end

            # Send to previous rank
            if rank > 0
                MPI.Send(u[1], comm; dest = rank - 1)
            end

            # Recieve from previous rank
            if rank > 0
                receive_buffer = Ref(0.0)
                MPI.Recv!(receive_buffer, comm; source = rank - 1)
                ghost_prev[] = receive_buffer[]
            end
        end

        function compute_task()
            u_copy[1] = 0.5 * (ghost_prev[] + u[2])
            u_copy[L] = 0.5 * (u[L - 1] + ghost_next[])

            for i in 2:L-1
                u_copy[i] = 0.5 * (u[i-1] + u[i+1])
            end
        end

        function compute_max_diff()
            max_diff = 0.0

            diff = abs(ghost_prev[] - u[1])
            max_diff = max(max_diff, diff)

            diff = abs(ghost_next[] - u[end])
            max_diff = max(max_diff, diff)

            for i in 2:L-1
                diff = abs(u[i] - u[i-1])
                max_diff = max(max_diff, diff)
            end

            max_diff
        end

        list_sig_terminate = Array{Bool}(undef, P)
        n_iterations = 0

        # Step 3: Run iterations
        while true
            # Step 3.1: Terminate if maximum difference is less than tolerance
            max_diff = compute_max_diff()
            sig_terminate = max_diff < tolerance
            MPI.Allgather!([sig_terminate], list_sig_terminate, comm)

            if all(list_sig_terminate)
                break
            end

            global n_iterations += 1

            # Step 3.2: Exchange ghost variables
            exchange_ghosts()

            # Step 3.3: Compute array
            compute_task()
            copy!(u, u_copy)
        end

        # Step 4: Gather results
        result = zeros(N)
        MPI.Gather!(u, result, comm; root)
        result = [boundary_values[1]; result[:]; boundary_values[2]]

        if rank == root
            @show N, P, n_iterations, tolerance, compute_max_diff()
        end
    end
end

run(`$(mpiexec()) -np 5 julia --project=. -e $code`);
