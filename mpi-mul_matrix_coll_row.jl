# Algorithm to compute matrix matrix multiplication by algorithm 3 and using MPI Collectives (Row-wise N/P)

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

            # Step 1: Send size(B) to workers
            if rank == 0
                myN = size(A, 1)
                Nref = Ref(myN)
            else
                Nref = Ref(0)
            end
            MPI.Bcast!(Nref, comm; root)
            
            myN = Nref[]

            # Step 2: Send B to workers
            if rank == 0
                myB = B
            else
                myB = zeros(N, N)
            end
            MPI.Bcast!(myB, comm; root)

            # Step 3: Transfer Ai to workers
            L = div(myN, P)
            myAt = zeros(L, myN)
            At = transpose(A)
            MPI.Scatter!(At, myAt, comm; root)
            myA = transpose(myAt)

            # Step 4: Compute myAt*myB
            myC = myAt * myB

            # Step 5: Communicate result back to root
            myCt = transpose(myC)
            Ct = similar(C)
            MPI.Gather!(myCt, Ct, comm; root)
            C .= transpose(Ct)
        end

        N_ = 1000
        A_ = rand(N_, N_)
        B_ = rand(N_, N_)
        C_ = zeros(N_, N_)

        @time matrix_mul!(C_, A_, B_)

        if rank == 0
            if C_ ≈ A_ * B_
                println("Great success!")
            else
                println("Nope, something went wrong!")
            end
        end
    end
end

run(`$(mpiexec()) -np 11 julia --project=. -e $code`)
