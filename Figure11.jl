#HR edit
import DPFEHM
import GaussianRandomFields
import Random
import BSON
import PyPlot
import StatsBase
import Flux
import LaTeXStrings



# Pressure histogram
dict=BSON.load("model_prediction_final3D.bson")
data1=dict[:pressure_vals1]
data2=dict[:extraction_rates1]
data3=dict[:pressure_vals2]
data4=dict[:extraction_rates2]


p50p = StatsBase.percentile(data1, 50)
p10p = StatsBase.percentile(data1, 10)
p90p = StatsBase.percentile(data1, 90)
fig, ax = PyPlot.subplots(dpi=1200)
ax.hist(data1; bins=100, edgecolor="black", facecolor="darkblue")
# ax.axvline(p10p, color="green", linestyle="--", label="10th percentile")
# ax.axvline(p90p, color="red", linestyle="--", label="90th percentile")
PyPlot.xlim(-0.0005, 0.0005)
ax.set_xlabel("Pressure at the critical location, MPa",
               fontsize=15, fontname="Arial")
ax.set_ylabel("Frequency", fontsize=15, fontname="Arial")
# Tick labels
for lbl in ax.get_xticklabels();  lbl.set_fontname("Arial"); lbl.set_fontsize(15); end
for lbl in ax.get_yticklabels();  lbl.set_fontname("Arial");  lbl.set_fontsize(15); end
# ax.legend(loc="upper left",fontsize=12)
display(fig)
fig.savefig("Figure11a.pdf", dpi=1200)
close(fig)


fig, ax = PyPlot.subplots(dpi=1200)
ax.hist(data3; bins=100, edgecolor="black", facecolor="darkblue")
# ax.axvline(p10p, color="green", linestyle="--", label="10th percentile")
# ax.axvline(p90p, color="red", linestyle="--", label="90th percentile")
PyPlot.xlim(-0.0005, 0.0005)
ax.set_xlabel("Pressure at the critical location, MPa",
               fontsize=15, fontname="Arial")
ax.set_ylabel("Frequency", fontsize=15, fontname="Arial")
# Tick labels
for lbl in ax.get_xticklabels();  lbl.set_fontname("Arial"); lbl.set_fontsize(15); end
for lbl in ax.get_yticklabels();  lbl.set_fontname("Arial");  lbl.set_fontsize(15); end
# ax.legend(loc="upper left",fontsize=12)
display(fig)
fig.savefig("Figure11b.pdf", dpi=1200)
close(fig)



# @BSON.load "mytrained_model_FinalTrainMPILD3D.bson" model


# global losses_train = Float64[]
# global losses_test = Float64[]
# global rmses_train = Float64[]
# global rmses_test = Float64[]
# global train_time = Float64[]

# global pressure_vals1 = Float64[]
# global extraction_rates1= Float64[]

# global pressure_vals2 = Float64[]
# global extraction_rates2= Float64[]

# mutable struct Fluid_n
#     vw::Float64
#     vo::Float64
#     swc::Float64
#     sor::Float64
# end

# Random.seed!(1234) #for paper plots
# # Injection rate
# # Injection rate
# Qinj = [0.031688, 0.031688] # [m^3/s] (1 MMT water/yr)
# n = 26
# ns = [n, n,3]
# steadyhead = 0e0
# sidelength = 500
# thickness  = 1.0
# mins = [-sidelength, -sidelength, 0] #meters
# maxs = [sidelength, sidelength, 6] #meters
# num_eigenvectors = 200
# sigma = 1.0
# lambda = 50
# cov = GaussianRandomFields.CovarianceFunction(2, GaussianRandomFields.Matern(lambda, 1; σ=sigma))
# x_pts = range(mins[1], maxs[1]; length=ns[1])
# y_pts = range(mins[2], maxs[2]; length=ns[2])
# grf = GaussianRandomFields.GaussianRandomField(cov, GaussianRandomFields.KarhunenLoeve(num_eigenvectors), x_pts, y_pts)

# pressure_target  = 0
# learning_rate = 1e-4
# coords, neighbors, areasoverlengths, volumes=DPFEHM.regulargrid3d(mins, maxs, ns);#build the grid
# dirichletnodes = Int[]
# dirichleths = zeros(size(coords, 2))
# monitoring_well_node = [201*3-1 189*3-1]
# injection_extraction_nodes = [267*3-1, 279*3-1, 487*3-1, 475*3-1]
# @show coords[:,injection_extraction_nodes]
# @show coords[:,monitoring_well_node]
# h0 = zeros(size(coords, 2))
# fluid=Fluid_n(1.0, 1.0, 0.0, 0.0)
# S0=zeros(size(coords, 2))
# nt = 45;  dt =8*24*60*60;

# for i = 1:size(coords, 2)
#     if abs(coords[1, i]) == sidelength || abs(coords[2, i]) == sidelength
#         push!(dirichletnodes, i)
#         dirichleths[i] = steadyhead
#     end
# end

# function getQs(Qs, is)
#     sum(Qs .* ((collect(1:size(coords, 2)) .== i) for i in is))
# end
# function solve_numerical(Qs, T)
#     Qs = getQs(Qs, injection_extraction_nodes)
#     everystep=false # output all the time steps
#     Ts=exp.(reshape(T,size(Qs)))
#     args=h0, S0, Ts, dirichleths,  dirichletnodes, Qs, volumes, areasoverlengths, fluid, dt, neighbors, nt, everystep
#     h_gw_t, S= DPFEHM.solvetwophase(args...)
#     # push!(pressure_vals, h_gw_t[monitoring_well_node] - steadyhead)
#     # push!(extraction_rates, Qs[injection_extraction_nodes[1]])
#     return h_gw_t
# end



# function attach_injection(Q1, Qinj)
#     # Q1 is (2, N), Qinj is length-2
#     N = size(Q1, 2)
#     inj = repeat(reshape(Qinj, :, 1), 1, N)   # (2, N)
#     return vcat(Q1, inj)                      # (4, N)
# end


# nsample=100

# for i=1:nsample
#     @show i
#     T = repeat(reshape(GaussianRandomFields.sample(grf)', 1, ns[2], ns[1]), ns[3], 1, 1)
#     Ts=(reshape(T,size(coords,2)))
#     Tst=reshape(Ts,n,n,3,1)
#     Q1 = model(Tst)   
#     Qs = attach_injection(Q1, Qinj)              # (4,N)
#     qi  = view(Qs, :, 1)                     # length-4
#     Ti  = view(Ts, :, :, :, 1)               # (26,26,3)

#     h_gw_t=solve_numerical(qi, Ti)

#     push!(pressure_vals1, h_gw_t[monitoring_well_node[1]] - steadyhead)
#     push!(extraction_rates1, Q1[1])
    
#     push!(pressure_vals2, h_gw_t[monitoring_well_node[2]] - steadyhead)
#     push!(extraction_rates1, Q1[2])

# end



# data1=pressure_vals1
# data2=extraction_rates1

# data3=pressure_vals2
# data4=extraction_rates2
# @BSON.save "model_prediction_final3Dt.bson"  pressure_vals1 extraction_rates1 pressure_vals2 extraction_rates2