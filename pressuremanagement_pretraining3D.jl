#HR edit
import DPFEHM
import GaussianRandomFields
import Optim
import Random
import BSON
import Flux

losses_train_pt = Float64[]
losses_test_pt = Float64[]
rmses_train_pt = Float64[]
rmses_test_pt = Float64[]
train_time_pt = Float64[]




# Injection rate
Qinj = [0.031688, 0.031688] # [m^3/s] (1 MMT water/yr)
n = 26
ns = [n, n,3]
steadyhead = 0e0
sidelength = 500
thickness  = 1.0
mins = [-sidelength, -sidelength, 0] #meters
maxs = [sidelength, sidelength, 3] #meters
num_eigenvectors = 200
sigma = 1.0
lambda = 50
cov = GaussianRandomFields.CovarianceFunction(2, GaussianRandomFields.Matern(lambda, 1; σ=sigma))
x_pts = range(mins[1], maxs[1]; length=ns[1])
y_pts = range(mins[2], maxs[2]; length=ns[2])
grf = GaussianRandomFields.GaussianRandomField(cov, GaussianRandomFields.KarhunenLoeve(num_eigenvectors), x_pts, y_pts)

pressure_target  = 0
learning_rate = 1e-4
coords, neighbors, areasoverlengths, volumes=DPFEHM.regulargrid3d(mins, maxs, ns);#build the grid
dirichletnodes = Int[]
dirichleths = zeros(size(coords, 2))
monitoring_well_nodes = [190*3 200*3]
injection_extraction_nodes = [271*3, 280*3, 487*3, 475*3]


for i = 1:size(coords, 2)
    if abs(coords[1, i]) == sidelength || abs(coords[2, i]) == sidelength
        push!(dirichletnodes, i)
        dirichleths[i] = steadyhead
    end
end

function getQs(Qs, is)
    sum(Qs .* ((collect(1:size(coords, 2)) .== i) for i in is))
end

# function solve_numerical(Qs, T)
#     permInv=(exp.(T)).^(-1)
#     logKs2Ks_neighbors(Ks) = 2*( (Ks[map(p->p[1], neighbors)] .+ Ks[map(p->p[2], neighbors)]).^(-1)) # hermonic average
#     Qs = getQs(Qs, injection_extraction_nodes)
#     Ks_neighbors = logKs2Ks_neighbors(permInv)
#     h_gw = DPFEHM.groundwater_steadystate(Ks_neighbors, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs)
#     return h_gw[monitoring_well_node]- steadyhead
# end
# function getQs(qvals::AbstractVector, idxs::AbstractVector{<:Integer})
#     Q = zeros(Float64, size(coords, 2))  # one entry per node
#     @inbounds for j in 1:length(idxs)
#         Q[idxs[j]] = qvals[j]
#     end
#     return Q
# end
function solve_numerical(q4::AbstractVector, T)
    # T is a (26,26,3) field after your reshape
    permInv = (exp.(T)).^(-1)
    logKs2Ks_neighbors(Ks) = 2 * ((Ks[map(p->p[1], neighbors)] .+ Ks[map(p->p[2], neighbors)]).^(-1)) # harmonic avg
    Qs = getQs(q4, injection_extraction_nodes)   # <— pass 4-vector here
    Ks_neighbors = logKs2Ks_neighbors(permInv)
    h_gw = DPFEHM.groundwater_steadystate(Ks_neighbors, neighbors, areasoverlengths,
                                          dirichletnodes, dirichleths, Qs)
    return sum(h_gw[monitoring_well_nodes]) - steadyhead
end

#like LeNet5
# model = Flux.Chain(Flux.Conv((5, 5), 1=>6, Flux.relu),
#               Flux.MaxPool((2, 2)),
#               Flux.Conv((5, 5), 6=>16, Flux.relu),
#               Flux.MaxPool((2, 2)),
#               Flux.flatten,
#               Flux.Dense(144, 120, Flux.relu),
#               Flux.Dense(120, 84, Flux.relu),
#               Flux.Dense(84, 2)) |> Flux.f64
model = Flux.Chain(
    Flux.Conv((5, 5), 3 => 6, Flux.relu),  # <-- 3 input channels now
    Flux.MaxPool((2, 2)),             # 26→22→11
    Flux.Conv((5, 5), 6 => 16, Flux.relu), # 11→7
    Flux.MaxPool((2, 2)),             # 7→3
    Flux.flatten,                     # 16 * 3 * 3 = 144
    Flux.Dense(144, 120, Flux.relu),
    Flux.Dense(120, 84, Flux.relu),
    Flux.Dense(84, 2)
)|> Flux.f64


# Make neural network parameters trackable by Flux
θ = Flux.params(model)

# function loss(model,x)
#     Ts = reshape(hcat(map(y->y[1], x)...), size(x[1][1], 1), size(x[1][1], 2), 1, length(x))
#     targets = map(y->y[2], x)
#     Q1 = model(Ts)
#     Qs = map(Q->[Q, Qinj], Q1)
#     loss = sum(map(i->solve_numerical(Qs[i], Ts[:, :, 1, i]) - targets[i], 1:size(Ts, 4)).^2)
#     return loss
# end



# function loss(model, x)
#     # Collect the 3D input fields (3×26×26) from the batch
#     fields = map(y -> y[1], x)             # each is (3,26,26)
#     nfields = length(fields)

#     # Stack along the 4th dimension (batch)
#     # After permutedims: (26,26,3), so cat(..., dims=4) → (26,26,3,N)
#     Ts = cat((permutedims(f, (2,3,1)) for f in fields)...; dims=4)
#     # @show size(Ts)
#     # Targets (scalar pressure values)
#     targets = map(y -> y[2], x)

#     # Forward pass
#     Q1 = model(Ts)
#     # @show Qinj[1]

#     # Attach injection rate if your pipeline needs it
#     N = size(Q1, 2)
#     Qs = [ [Q1[1,i], Q1[2,i], Qinj[1], Qinj[2]] for i in 1:N ]
    
#     # Physics loss
#     loss_val = sum(map(i -> solve_numerical(Qs[i], Ts[:, :, :, i]) - targets[i],
#                        1:nfields).^2)

#     return loss_val
# end
# function stack_fields_to_batch(fields::Vector{<:AbstractArray})
#     # fields are each (3,26,26)  -> we want (26,26,3,N)
#     H, W, C = 26, 26, 3
#     N = length(fields)
#     Ts = Array{Float32}(undef, H, W, C, N)
#     @inbounds for i in 1:N
#         # Move (3,26,26) -> (26,26,3)
#         Ts[:,:,:,i] = permutedims(fields[i], (2,3,1))
#     end
#     return Ts
# end
function stack_fields_to_batch(fields::Vector{<:AbstractArray})
    mapped = map(f -> permutedims(f, (2,3,1)), fields)   # each → (26,26,3)
    Ts = cat(mapped...; dims=4)                          # (26,26,3,N)
    return Ts
end
function attach_injection(Q1::AbstractMatrix, Qinj::AbstractVector)
    # Q1 is (2, N), Qinj is length-2
    N = size(Q1, 2)
    inj = repeat(reshape(Qinj, :, 1), 1, N)   # (2, N)
    return vcat(Q1, inj)                      # (4, N)
end
function loss(model, x)
    # x is the batch: Vector{Tuple(field, target)} with field (3,26,26)
    fields  = map(y -> y[1], x)
    targets = map(y -> y[2], x)
    N       = length(fields)

    Ts = stack_fields_to_batch(fields)           # (26,26,3,N), Float32
    Q1 = model(Ts)                               # (2,N)
    Qs = attach_injection(Q1, Qinj)              # (4,N)

    # Physics residual for each sample, then sum of squares
    acc = 0.0
    @inbounds for i in 1:N
        # Take the i-th 4-vector of rates and the i-th field
        qi  = view(Qs, :, i)                     # length-4
        Ti  = view(Ts, :, :, :, i)               # (26,26,3)
        r   = solve_numerical(qi, Ti) - targets[i]
        acc += r^2
    end
    return acc
end

# opt = ADAM(learning_rate)
opt_state = Flux.setup(Flux.Adam(learning_rate), model)  
# Training epochs
epochs = 1:500
batch_size = 40



data_train_batch = [[(repeat(reshape(GaussianRandomFields.sample(grf)', 1, ns[2], ns[1]), ns[3], 1, 1), pressure_target) for i = 1:batch_size] for v in 1:200/batch_size]
data_test = [[(repeat(reshape(GaussianRandomFields.sample(grf)', 1, ns[2], ns[1]), ns[3], 1, 1), pressure_target) for i = 1:batch_size] for i = 1:200/batch_size]
println("The training has started..")
loss_train = sum(map(x->loss(model,x), data_train_batch))
rmse_train = sqrt(loss_train/(batch_size *length(data_train_batch)))
loss_test = sum(map(x->loss(model,x), data_test))
rmse_test = sqrt(loss_test/(batch_size *length(data_test)))
println(string("epoch: 0 train rmse: ", rmse_train, " test rmse: ", rmse_test))
push!(losses_test_pt, loss_test)
push!(rmses_test_pt, rmse_test)
push!(rmses_train_pt, rmse_train)

for epoch in epochs
    data_train_batch = [[(repeat(reshape(GaussianRandomFields.sample(grf)', 1, ns[2], ns[1]), ns[3], 1, 1), pressure_target) for i = 1:batch_size] for v in 1:200/batch_size]
    tt = @elapsed Flux.train!(loss, model, data_train_batch, opt_state)
    push!(train_time_pt, tt)
    loss_train = sum(map(x->loss(model,x), data_train_batch))
    rmse_train = sqrt(loss_train/(batch_size *length(data_train_batch)))
    loss_test = sum(map(x->loss(model,x), data_test))
    rmse_test = sqrt(loss_test/(batch_size *length(data_test)))

    # Terminal output
    println(string("epoch: ", epoch, " time: ", tt, " train rmse: ", rmse_train, " test rmse: ", rmse_test))
    push!(losses_train_pt, loss_train)
    push!(rmses_train_pt, rmse_train)
    push!(losses_test_pt, loss_test)
    push!(rmses_test_pt, rmse_test)
end
@BSON.save "model_preTrainingLD_$(batch_size)_$(learning_rate)3D.bson"  epochs train_time_pt losses_train_pt  rmses_train_pt  losses_test_pt  rmses_test_pt  
@BSON.save "mytrained_model_preTrainLD3D.bson" model

