# Went wrong because I'm not treating ghost cells separately. I'm sending those as two extra elements of the array. 
# Didn't necessarily go wrong but the index calculation issue became so complex that it only works with N=6, P=3
# I'm out of brain capacity, so gonna try to handle ghost cells as just two simple variables.

using MPI

code = quote
    begin
        using MPI
        MPI.Init()

        comm = MPI.Comm_dup(MPI.COMM_WORLD)
        P = MPI.Comm_size(comm)
        rank = MPI.Comm_rank(comm)

        root = 0
        N = 6

        function isroot()
            root == rank
        end

        function compute_task!(myu, myu_copy)
            for i in 2:(size(myu, 1)-1)
                myu_copy[i] = (myu[i-1] + myu[i+1]) * 0.5
            end

            ghost_prev = myu_copy[2] # goes from Pi to Pi-1
            ghost_next = myu_copy[end-1] # goes from Pi to Pi+1

            (ghost_prev, ghost_next)
        end

        L = Int(N / P)

        u = [-1.0, 1.0, 2.0, 1.0, 3.0, 0.0, 1.0, 4.0]
        u_copy = copy(u)

        tolerance = 1e-5

        myu = zeros(L + 2) # L + 2 ghost cells
        myu_copy = zeros(L + 2)

        # Step 1: Communicate u to processes
        if isroot()
            myu .= u[1:L+2]
            myu_copy .= copy(myu)

            for Pi in 1:P-1
                start_idx = L * (Pi + 1) - 1
                end_idx = start_idx + L + 1
                MPI.Send(u[start_idx:end_idx], comm; dest=Pi)
            end
        else
            MPI.Recv!(myu, comm; source=root)
            myu_copy .= copy(myu)
        end

        max_iters = 100

        for iter in 1:max_iters
            # Step 2: Compute task
            ghost_prev, ghost_next = compute_task!(myu, myu_copy)
            copy!(myu, myu_copy)

            # Step 3: Communicate ghost cells
            if rank < P - 1
                # send to next
                MPI.Send(ghost_next, comm; dest=rank + 1)
            end

            if rank > 0
                # receive from prev
                x = Ref(0.0)
                MPI.Recv!(x, comm; source=rank - 1)
                myu[1] = x[]
            end

            if rank > 0
                # send to prev
                MPI.Send(ghost_prev, comm; dest=rank - 1)
            end

            if rank < P - 1
                # recieve from next
                x = Ref(0.0)
                MPI.Recv!(x, comm; source=rank + 1)
                myu[end] = x[]
            end
        end

        # Gather data from all arrays
        uans = zeros(N)

        if isroot()
            uans[1:L] .= myu[2:end-1]

            for Pi in 1:P-1
                recbuff = zeros(L)
                MPI.Recv!(recbuff, comm; source=Pi)

                # @show rank, Pi, recbuff
                uans[(Pi * L) + 1: (Pi * L) + L] .= recbuff[:]
            end
        else
            MPI.Send(myu[2:end-1], comm; dest=root)
        end

        if isroot()
            result = [u[1]; uans; u[end]]
            @show result
        end
    end
end

run(`$(mpiexec()) -np 3 julia --project=. -e $code`);
