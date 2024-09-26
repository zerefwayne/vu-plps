# Algorithm to compute matrix matrix multiplication by algorithm 3 and using MPI Collectives (Col-wise N/P)

using MPI

code = quote
    begin
        using MPI
        MPI.Init()

        comm = MPI.Comm_dup(MPI.COMM_WORLD)
        rank = MPI.Comm_rank(comm)
        P = MPI.Comm_size(comm)
        
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

            # Step 2: Send A to workers
            if rank == 0
                myA = A
            else
                myA = zeros(myN, myN)
            end
            MPI.Bcast!(myA, comm; root)

            # Step 3: Transfer Bi to workers
            L = div(myN, P)
            myB = zeros(myN, L)
            MPI.Scatter!(B, myB, comm; root)            
            
            # Step 4: Compute myA*myB
            myC = myA * myB

            # Step 5: Communicate result back to root
            MPI.Gather!(myC, C, comm; root)
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

run(`$(mpiexec()) -np 10 julia --project=. -e $code`);
