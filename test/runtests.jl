using Test
using TensorFactorizations
using TensorOperations
using Random
using ArgParse
using LinearAlgebra


function parse_pars()
    settings = ArgParseSettings(autofix_names=true)
    @add_arg_table(settings
        , "--iters", arg_type=Int, default=50
        , "--no_test_svd", action=:store_true
        , "--no_test_eig", action=:store_true
        , "--no_test_split", action=:store_true
    )
    pars = parse_args(ARGS, settings; as_symbols=true)
    pars[:test_svd] = !pars[:no_test_svd]
    pars[:test_eig] = !pars[:no_test_eig]
    pars[:test_split] = !pars[:no_test_split]
    return pars
end


""" Generate a random shape. """
function rshape(;n=nothing, chi=nothing, nlow=0, nhigh=5, chilow=1, chihigh=5)
    # n is the number of legs, chi is the bond dimension(s)
    n == nothing && (n = rand(nlow:nhigh))
    shape = (chi == nothing ? rand(chilow:chihigh, (n,))
             : collect(repeat([chi], n)))
    return shape 
end


""" Generate a random tensor, either with a given or a random shape. """
function rtensor(;shape=nothing, n=nothing, chi=nothing,
                 nlow=0, nhigh=5, chilow=0, chihigh=6, dtype=nothing)
    if dtype == nothing
        dtype = rand([Int, Float64, Complex{Float64}])
    end
    if shape == nothing
        shape = rshape(n=n, chi=chi, nlow=nlow, nhigh=nhigh, chilow=chilow,
                       chihigh=chihigh)
    end
    A = rand(dtype, (shape...),)
    return A
end


function test_svd()
    shp = rshape(nlow=2)
    n = length(shp)
    perm = randperm(n)
    I = rand(1:n-1)
    J = n - I
    ins = perm[1:I]
    outs = perm[I+1:end]
    A = rtensor(shape=shp)

    # SVD and reconstruct without truncation
    U, S, Vt = tensorsvd(A, ins, outs)
    US = tensorcontract(U, [-1*(1:I); 1], Diagonal(S), [1,-I-1])
    Areco = tensorcontract(US, [-1*(1:I); 1], Vt, [1; -1*(I+1:n)])
    Areco = tensorcopy(Areco, collect(1:ndims(Areco)), invperm(perm))
    @test isapprox(A, Areco)

    # SVD and reconstruct with truncation
    eps = 10.0^(-rand(0:3))
    dim_in = prod(shp[perm][1:I])
    dim_out = prod(shp[perm][I+1:end])
    max_dim = min(dim_in, dim_out)
    max_chi = rand(1:max_dim)
    chis = 1:max_chi
    U, S, Vt, error = tensorsvd(A, ins, outs, eps=eps, chis=chis,
                                return_error=true)
    # This test could fail if there were degenerate singular values. We assume
    # this is not the case because the tensors are random.
    @test (error<eps || length(S) == max_chi)
    US = tensorcontract(U, [-1*(1:I); 1], Diagonal(S), [1,-I-1])
    Areco = tensorcontract(US, [-1*(1:I); 1], Vt, [1; -1*(I+1:n)])
    Areco = tensorcopy(Areco, collect(1:ndims(Areco)), invperm(perm))
    Anorm = norm(A)
    true_error = Anorm > 0 ? norm(Areco - A)/Anorm : norm(Areco)
    @test isapprox(error, true_error; atol=1e-8)
end


function test_eig()
    # Non-hermitian
    I = rand(1:3)
    n = 2*I
    shp = rshape(n=n)
    perm = randperm(n)
    ins = perm[1:I]
    outs = perm[I+1:end]
    # Making sure that the left and right legs match.
    shp = collect(shp)
    for (o,i) in zip(outs, ins)
        shp[o] = shp[i]
    end
    shp = (shp...,)
    A = rtensor(shape=shp)

    # Check that eigenvectors really are eigenvectors
    eps = 10.0^(-rand(0:3))
    dim_in = prod(shp[perm][1:I])
    dim_out = prod(shp[perm][I+1:end])
    max_dim = min(dim_in, dim_out)
    max_chi = rand(1:max_dim)
    chis = 1:max_chi
    E, U, error = tensoreig(A, ins, outs, eps=eps, chis=chis,
                            return_error=true, break_degenerate=true)
    @test (error<eps || length(E) == max_chi)
    A_contract_list = collect(repeat([0], n))
    for (k,i) in enumerate(ins)
        A_contract_list[i] = -k
    end
    for (k,i) in enumerate(outs)
        A_contract_list[i] = k
    end
    AU = tensorcontract(A, A_contract_list, U, [1:I; -(n+1)],
                        [-1*(1:I); -(n+1)])
    UE = tensorcontract(U, [-1*(1:I); 1], Diagonal(E), [1; -(n+1)],
                        [-1*(1:I); -(n+1)])
    @test isapprox(AU, UE)

    # Hermitian
    I = rand(1:3)
    n = 2*I
    shp = rshape(n=n)
    perm = randperm(n)
    ins = perm[1:I]
    outs = perm[I+1:end]
    # Making sure that the left and right legs match.
    shp = collect(shp)
    for (o,i) in zip(outs, ins)
        shp[o] = shp[i]
    end
    shp = (shp...,)
    A = rtensor(shape=shp)
    # Make Hermitian
    hperm = collect(1:n)
    for (o,i) in zip(outs, ins)
        hperm[o], hperm[i] = hperm[i], hperm[o]
    end
    At = conj!(tensorcopy(A, collect(1:ndims(A)), hperm))
    A = (A + At)/2

    # Decompose and reconstruct without truncation
    E, U = tensoreig(A, ins, outs, hermitian=true)
    UE = tensorcontract(U, [-1*(1:I); 1], Diagonal(E), [1,-I-1])
    Areco = tensorcontract(UE, [-1*(1:I); 1], conj(U), [-1*(I+1:n); 1])
    Areco = tensorcopy(Areco, collect(1:ndims(Areco)), invperm(perm))

    # Decompose and reconstruct with truncation
    eps = 10.0^(-rand(0:3))
    dim_in = prod(shp[perm][1:I])
    dim_out = prod(shp[perm][I+1:end])
    max_dim = min(dim_in, dim_out)
    max_chi = rand(1:max_dim)
    chis = 1:max_chi
    E, U, error = tensoreig(A, ins, outs, eps=eps, chis=chis,
                            return_error=true, hermitian=true)
    # This test could fail if there were degenerate singular values. We assume
    # this is not the case because the tensors are random.
    @test (error<eps || length(E) == max_chi)
    UE = tensorcontract(U, [-1*(1:I); 1], Diagonal(E), [1,-I-1])
    Areco = tensorcontract(UE, [-1*(1:I); 1], conj(U), [-1*(I+1:n); 1])
    Areco = tensorcopy(Areco, collect(1:ndims(Areco)), invperm(perm))
    Anorm = norm(A)
    true_error = Anorm > 0 ? norm(Areco - A)/Anorm : norm(Areco)
    @test isapprox(error, true_error; atol=1e-8)
end


function test_split()
    # Non-hermitian, SVD
    shp = rshape(nlow=2)
    n = length(shp)
    perm = randperm(n)
    I = rand(1:n-1)
    J = n - I
    ins = perm[1:I]
    outs = perm[I+1:end]
    A = rtensor(shape=shp)

    B1, B2 = tensorsplit(A, ins, outs)
    U, S, Vt = tensorsvd(A, ins, outs)
    Ssqrt = Diagonal(sqrt.(S))
    B1reco = tensorcontract(U, [-1*(1:I); 1], Ssqrt, [1, -n-1])
    B2reco = tensorcontract(Ssqrt, [-1,1], Vt, [1; -1*(2:J+1)])
    @test isapprox(B1reco, B1)
    @test isapprox(B2reco, B2)

    # Hermitian, eigendecomposition
    I = rand(1:3)
    n = 2*I
    shp = rshape(n=n)
    perm = randperm(n)
    ins = perm[1:I]
    outs = perm[I+1:end]
    # Making sure that the left and right legs match.
    shp = collect(shp)
    for (o,i) in zip(outs, ins)
        shp[o] = shp[i]
    end
    shp = (shp...,)
    A = rtensor(shape=shp)
    # Make Hermitian
    hperm = collect(1:n)
    for (o,i) in zip(outs, ins)
        hperm[o], hperm[i] = hperm[i], hperm[o]
    end
    At = conj!(tensorcopy(A, collect(1:ndims(A)), hperm))
    A = (A + At)/2

    B1, B2 = tensorsplit(A, ins, outs, hermitian=true)
    E, U = tensoreig(A, ins, outs, hermitian=true)
    Esqrt = Diagonal(sqrt.(complex.(E)))
    B1reco = tensorcontract(U, [-1*(1:I); 1], Esqrt, [1, -n-1])
    B2reco = tensorcontract(Esqrt, [-1,1], conj(U), [-1*(2:I+1); 1])
    @test isapprox(B1reco, B1)
    @test isapprox(B2reco, B2)
end


function main()
    pars = parse_pars()
    if pars[:test_svd]
        println("Testing SVD.")
        for iter_num in 1:pars[:iters]
            test_svd()
        end
        println("Done testing SVD.")
    end
    if pars[:test_eig]
        println("Testing eig.")
        for iter_num in 1:pars[:iters]
            test_eig()
        end
        println("Done testing eig.")
    end
    if pars[:test_split]
        println("Testing split.")
        for iter_num in 1:pars[:iters]
            test_split()
        end
        println("Done testing split.")
    end
end

main()

