#HR edit
import DPFEHM
import GaussianRandomFields
import Optim
import Random
import BSON
import MPI
import Flux
import Zygote
import Optimisers
import Functors



MPI.Init()
comm   = MPI.COMM_WORLD
rank   = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

include("twoPhase.jl")


n = 26
ns = [n, n,3]
sidelength = 500.0
mins = [-sidelength, -sidelength, 0] #meters
maxs = [sidelength, sidelength, 6] #meters
thickness = 1.0

coords, neighbors, areasoverlengths, volumes=DPFEHM.regulargrid3d(mins, maxs, ns);#build the grid

steadyhead = 0.0
dirichletnodes = Int[]
dirichleths = zeros(size(coords, 2))
monitoring_well_nodes = [201*3-1 189*3-1]
injection_extraction_nodes = [267*3-1, 279*3-1, 487*3-1, 475*3-1]


for i = 1:size(coords, 2)
    if abs(coords[1, i]) == sidelength || abs(coords[2, i]) == sidelength
        push!(dirichletnodes, i)
        dirichleths[i] = steadyhead
    end
end

Qinj =  [0.031688, 0.031688] # [m^3/s] (1 MMT water/yr)
pressure_target = 0.0

h0 = zeros(size(coords,2))
S0 = zeros(size(coords,2))
mutable struct Fluid; vw; vo; swc; sor; end
fluid = Fluid(1.0, 1.0, 0.0, 0.0)

nt = 45;  dt=8*24*60*60;

pressure_target  = 0
learning_rate = 1e-4
num_eigenvectors = 200
sigma = 1.0
lambda = 50
cov = GaussianRandomFields.CovarianceFunction(2, GaussianRandomFields.Matern(lambda, 1; σ=sigma))
x_pts = range(mins[1], maxs[1]; length=ns[1])
y_pts = range(mins[2], maxs[2]; length=ns[2])
grf = GaussianRandomFields.GaussianRandomField(cov, GaussianRandomFields.KarhunenLoeve(num_eigenvectors), x_pts, y_pts)



@BSON.load "mytrained_model_preTrainLD3D.bson" model

θ = Flux.params(model)

batch_size  = 10
num_samples = 100
num_batches = div(num_samples, batch_size)

function getQs(qs, nodes)
  sum(qs .* ((collect(1:size(coords,2)) .== i)
             for i in nodes))
end

function solve_numerical(qs, T)
    # your code from before:
    Qs_full = getQs(qs, injection_extraction_nodes)
    permVal  = exp.(reshape(T,length(T)))
    P_last, _ = solvetwophase(
    h0, S0, permVal,
    dirichleths, dirichletnodes,
    Qs_full, volumes, areasoverlengths,
    fluid, dt, neighbors, nt, false
    )
    return sum(P_last[monitoring_well_nodes]) - steadyhead
end
# MPI helpers
function local_args(args::Vector, r::Int, np::Int)
  N = length(args)
  base, rem = divrem(N, np)
  counts = [i < rem ? base+1 : base for i in 0:np-1]
  offs = cumsum([0; counts[1:end-1]])
  lo, hi = offs[r+1]+1, offs[r+1]+counts[r+1]
  return args[lo:hi]
end

function allreduce_grads!(grads)
  for p in θ
    g = grads[p]
    buf = copy(g)
    MPI.Allreduce!(buf, g, MPI.SUM, comm)
    g .*= 1/nprocs
  end
end

# ------------------------------------------------------------
function loss_local(model,batch)
  local_batch = local_args(batch, rank, nprocs)

  # build 4D input—with a Vector comprehension, not a Generator
  slices = [ reshape(x[1], ns[1], ns[2], ns[3]) for x in local_batch ]
  Ts_arr = cat(slices...; dims=4)

  Q_preds = model(Ts_arr)
  # purely functional map—no push! no Generator.
  Ts_list = first.(local_batch)
  targs   = last.(local_batch)
  errs = map(
        (q, T, targ) -> solve_numerical((q[1], q[2], Qinj[1], Qinj[2]), T) - targ,
        eachcol(Q_preds), Ts_list, targs
  )

  return sum(abs2, errs)
end


# function loss(model, x)
#     # x is the batch: Vector{Tuple(field, target)} with field (3,26,26)
#     fields  = map(y -> y[1], x)
#     targets = map(y -> y[2], x)
#     N       = length(fields)

#     Ts = stack_fields_to_batch(fields)           # (26,26,3,N), Float32
#     Q1 = model(Ts)                               # (2,N)
#     Qs = attach_injection(Q1, Qinj)              # (4,N)

#     # Physics residual for each sample, then sum of squares
#     acc = 0.0
#     @inbounds for i in 1:N
#         # Take the i-th 4-vector of rates and the i-th field
#         qi  = view(Qs, :, i)                     # length-4
#         Ti  = view(Ts, :, :, :, i)               # (26,26,3)
#         r   = solve_numerical(qi, Ti) - targets[i]
#         acc += r^2
#     end
#     return acc
# end
# ------------------------------------------------------------
# θ  = Flux.params(model)
# opt = Flux.ADAM(1e-4)

opt_state = Optimisers.setup(Optimisers.Adam(1e-4), model)

epochs = 1:200
rmses_train = Float64[]
rmses_test  = Float64[]
train_time= Float64[]
# data_train = [[(GaussianRandomFields.sample(grf), pressure_target) for i = 1:batch_size] for v in 1:200/batch_size]
data_test = [[(repeat(reshape(GaussianRandomFields.sample(grf)', 1, ns[2], ns[1]), ns[3], 1, 1), pressure_target) for i = 1:batch_size] for i = 1:num_samples/batch_size]
data_train = [[(repeat(reshape(GaussianRandomFields.sample(grf)', 1, ns[2], ns[1]), ns[3], 1, 1), pressure_target) for i = 1:batch_size] for v in 1:num_samples/batch_size]


local_train_loss  = sum(loss_local(model,b) for b in data_train)
global_train_loss = MPI.Allreduce(local_train_loss, MPI.SUM, comm)
rmse_train = sqrt(global_train_loss / num_samples)

local_test_loss  = sum(loss_local(model,b) for b in data_test)
global_test_loss = MPI.Allreduce(local_test_loss, MPI.SUM, comm)
rmse_test = sqrt(global_test_loss / num_samples)

if rank == 0
  println(string("epoch: 0  train rmse: ", rmse_train, " test rmse: ", rmse_test))
  push!(rmses_train, rmse_train)
  push!(rmses_test,  rmse_test)

end

for epoch in epochs

    data_train = [[(repeat(reshape(GaussianRandomFields.sample(grf)', 1, ns[2], ns[1]), ns[3], 1, 1), pressure_target) for i = 1:batch_size] for v in 1:num_samples/batch_size]
    local_train_loss = 0.0
    MPI.Barrier(comm)
    t0 = MPI.Wtime()
    for batch in data_train
        # grads = Zygote.gradient(() -> loss_local(model,batch), θ)
      grads = Flux.gradient(model) do m
          loss_local(m, batch)
      end

      Functors.fmap(grads) do leaf
          leaf === nothing && return nothing
          buf = copy(leaf)
          MPI.Allreduce!(buf, leaf, MPI.SUM, comm)
          leaf ./= nprocs
          leaf
      end
        # Flux.Optimise.update!(opt_state, θ, grads)
        Optimisers.update!(opt_state, model, grads[1])
        local_train_loss += loss_local(model,batch)
    end
    MPI.Barrier(comm)
    t1 = MPI.Wtime()


    global_train_loss = MPI.Allreduce(local_train_loss, MPI.SUM, comm)
    rmse_train = sqrt(global_train_loss / num_samples)

    # compute & all‑reduce test loss

    local_test_loss  = sum(loss_local(model,b) for b in data_test)
    global_test_loss = MPI.Allreduce(local_test_loss, MPI.SUM, comm)
    rmse_test = sqrt(global_test_loss / num_samples)

    if rank == 0
        tt=t1-t0
        @BSON.save "mytrained_model_FinalTrainMPILD3D.bson" model
        push!(rmses_train, rmse_train)
        push!(rmses_test,  rmse_test)
        push!(train_time,  tt)
        println(string("epoch: ", epoch," time: ",tt , " train rmse: ", rmse_train, " test rmse: ", rmse_test))
    end
end
@BSON.save "mytrained_model_Finetuned3D.bson"  epochs  rmses_train  rmses_test 
MPI.Barrier(comm)
MPI.Finalize()
