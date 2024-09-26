using MPI

code = quote
    begin
        using MPI
        MPI.Init()

        comm = MPI.Comm_dup(MPI.COMM_WORLD)
        rank = MPI.Comm_rank(comm)

        P = MPI.Comm_size(comm) - 1
        root = 0

        function matrix_mul!(C, A, B)
            @assert size(A, 2) == size(B, 1)
            @assert size(A, 1) == size(C, 1)
            @assert size(B, 2) == size(C, 2)

            myN = size(A, 1)

            # Step 1: Transfer B to workers
            myB = B
            if rank == 0
                for Pi in 1:P
                    MPI.Send(B, comm; dest=Pi)
                end
            else
                status = MPI.Probe(comm, MPI.Status; source=root)
                count = MPI.Get_count(status, eltype(B))
                myN = Int(sqrt(count))
                myB = zeros(myN, myN)
                MPI.Recv!(myB, comm; source=root)
            end

            # Step 2: Transfer Ai to workers
            L = div(myN, P)
            myA = zeros(L, myN)
            if rank == 0
                for Pi in 1:P
                    lb = L * (Pi - 1) + 1
                    ub = L * Pi
                    MPI.Send(view(A, lb:ub, :), comm; dest=Pi)
                end
            else
                MPI.Recv!(myA, comm; source=root)
            end

            # # Step 3: Compute Ai * B return
            myC = myA * myB
            if rank == 0
                for Pi in 1:P
                    lb = L * (Pi - 1) + 1
                    ub = L * Pi
                    MPI.Recv!(view(C, lb:ub, :), comm; source=Pi)
                end
            else
                MPI.Send(myC, comm; dest=root)
            end
        end

        N_ = 1000
        A_ = rand(N_, N_)
        B_ = rand(N_, N_)
        C_ = zeros(N_, N_)

        @time matrix_mul!(C_, A_, B_)

        if rank == 0
            @assert C_ ≈ A_ * B_
        end
    end
end

run(`$(mpiexec()) -np 11 julia --project=. -e $code`)
