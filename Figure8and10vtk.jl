import DPFEHM
import GaussianRandomFields
import Random
import BSON
import PyPlot
import Flux


@BSON.load "mytrained_model_FinalTrainMPILD3D.bson" model
# @BSON.load "mytrained_model_preTrainLD3D.bson" model

mutable struct Fluid_n
    vw::Float64
    vo::Float64
    swc::Float64
    sor::Float64
end

# Random.seed!(1234) #for paper plots #Case 1
# Random.seed!(4567) #for paper plots #Case 2
Random.seed!(666555) #for paper plots #Case 2
# Injection rate
# Injection rate
Qinj = [0.031688, 0.031688] # [m^3/s] (1 MMT water/yr)
n = 26
ns = [n, n,3]
steadyhead = 0e0
sidelength = 500
thickness  = 1.0
mins = [-sidelength, -sidelength, 0] #meters
maxs = [sidelength, sidelength, 6] #meters
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
monitoring_well_node = [201*3-1 189*3-1]
injection_extraction_nodes = [267*3-1, 279*3-1, 487*3-1, 475*3-1]
@show coords[:,injection_extraction_nodes]
@show coords[:,monitoring_well_node]
h0 = zeros(size(coords, 2))
fluid=Fluid_n(1.0, 1.0, 0.0, 0.0)
S0=zeros(size(coords, 2))
nt = 45;  dt =8*24*60*60;
for i = 1:size(coords, 2)
    if abs(coords[1, i]) == sidelength || abs(coords[2, i]) == sidelength
        push!(dirichletnodes, i)
        dirichleths[i] = steadyhead
    end
end

function getQs(Qs, is)
    sum(Qs .* ((collect(1:size(coords, 2)) .== i) for i in is))
end

# function solve_numerical(q4, T)
#     # T is a (26,26,3) field after your reshape
#     permInv = (exp.(T)).^(-1)
#     logKs2Ks_neighbors(Ks) = 2 * ((Ks[map(p->p[1], neighbors)] .+ Ks[map(p->p[2], neighbors)]).^(-1)) # harmonic avg
#     Qs = getQs(q4, injection_extraction_nodes)   # <— pass 4-vector here
#     Ks_neighbors = logKs2Ks_neighbors(permInv)
#     h_gw = DPFEHM.groundwater_steadystate(Ks_neighbors, neighbors, areasoverlengths,
#                                           dirichletnodes, dirichleths, Qs)
#     return h_gw
# end
function solve_numerical(Qs, T)
    Qs = getQs(Qs, injection_extraction_nodes)
    everystep=false # output all the time steps
    Ts=exp.(reshape(T,size(Qs)))
    args=h0, S0, Ts, dirichleths,  dirichletnodes, Qs, volumes, areasoverlengths, fluid, dt, neighbors, nt, everystep
    h_gw_t, S= DPFEHM.solvetwophase(args...)
    # push!(pressure_vals, h_gw_t[monitoring_well_node] - steadyhead)
    # push!(extraction_rates, Qs[injection_extraction_nodes[1]])
    return h_gw_t
end



function attach_injection(Q1, Qinj)
    # Q1 is (2, N), Qinj is length-2
    N = size(Q1, 2)
    inj = repeat(reshape(Qinj, :, 1), 1, N)   # (2, N)
    return vcat(Q1, inj)                      # (4, N)
end


T = repeat(reshape(GaussianRandomFields.sample(grf)', 1, ns[2], ns[1]), ns[3], 1, 1)
Ts=(reshape(T,size(coords,2)))
Tst=reshape(Ts,n,n,3,1)

Q1 = model(Tst)   
# Q1=[0
#     0]                            # (2,N)
Qs = attach_injection(Q1, Qinj)              # (4,N)

qi  = view(Qs, :, 1)                     # length-4
Ti  = view(Ts, :, :, :, 1)               # (26,26,3)


h_gw_t=solve_numerical(qi, Ti)


# --- Build marker arrays on POINTS (same length as coords / h_gw_t) ---
npts = size(coords, 2)

# Flattened integer masks (0 everywhere)
markers      = zeros(Int8, npts)   # 0=none, 1=critical, 2=extraction, 3=injection
isCritical   = zeros(Int8, npts)   # separate boolean-style masks (0/1)
isInjection  = zeros(Int8, npts)
isExtraction = zeros(Int8, npts)

# Your special nodes (make sure these are linear point indices):
crit_nodes  = vec(monitoring_well_node)           # e.g. [570, 600] or [node_id]
injext      = vec(injection_extraction_nodes)     # e.g. [813, 840, 1461, 1425]
inj_nodes   = injection_extraction_nodes[3:end]                   # take your convention: even = injection
ext_nodes   = injection_extraction_nodes[1:2]                     # odd  = extraction

# Fill the masks
for i in crit_nodes
    isCritical[i] = 1
    markers[i]    = 1
end
for i in ext_nodes
    isExtraction[i] = 1
    markers[i]      = 2
end
for i in inj_nodes
    isInjection[i] = 1
    markers[i]     = 3
end

# Optional: a point ID array for labeling in ParaView
pointIDs = collect(1:npts)

# Reshape masks to point-data shape (nx, ny, nz) used for Pressure
markers_rs      = reshape(markers, reverse(ns)...)
isCritical_rs   = reshape(isCritical, reverse(ns)...)
isInjection_rs  = reshape(isInjection, reverse(ns)...)
isExtraction_rs = reshape(isExtraction, reverse(ns)...)
pointIDs_rs     = reshape(pointIDs, reverse(ns)...)

using WriteVTK
using Printf 

x=reshape(coords[1,:], reverse(ns)...)
y=reshape(coords[2,:], reverse(ns)...)
z=reshape(coords[3,:], reverse(ns)...)

filename = "FinalPressureFinetuned"
pres = reshape(h_gw_t, reverse(ns)...)

vtk_grid(filename, x, y, z) do vtk
    vtk["Pressure"] = pres
    vtk["Perm"] = (exp.(Ts)).^(-1)
end


# @show h_gw_t[monitoring_well_node]


# fig, ax = PyPlot.subplots()
# xmin, xmax = -sidelength, sidelength
# ymin, ymax = -sidelength, sidelength

# x = range(xmin, xmax, length=n)
# y = range(ymin, ymax, length=n)

# # Plot using imshow with correct spatial coordinates using extent
# img = ax.imshow(
#     reshape(Ts, n, n),
#     origin="lower",
#     cmap="viridis",
#     extent=[xmin, xmax, ymin, ymax],
#     aspect="equal",  # or use "equal"
#     interpolation="bicubic"
# )

# cb = fig.colorbar(img)
# cb.ax.tick_params(labelsize=12)  # Adjust 12 to your desired font size
# cb.set_label("Permeability (m²)", fontsize=14)

# # Cell numbers to mark
# monit = monitoring_well_node
# ext = injection_extraction_nodes[1]
# Inj = injection_extraction_nodes[2]

# # Convert linear indices to (row, column)
# row_monit = div(monit - 1, n) + 1
# col_monit = mod(monit - 1, n) + 1

# row_ext = div(ext - 1, n) + 1
# col_ext = mod(ext - 1, n) + 1

# row_Inj = div(Inj - 1, n) + 1
# col_Inj = mod(Inj - 1, n) + 1

# # Convert indices to coordinates
# x_monit = x[col_monit]
# y_monit = y[row_monit]

# x_ext = x[col_ext]
# y_ext = y[row_ext]

# x_Inj = x[col_Inj]
# y_Inj = y[row_Inj]

# offset = sidelength * 0.05  # adjust as needed for spacing

# # Marker for Monitoring Well
# ax.plot(x_monit, y_monit, marker="o", color="red", markersize=10, markeredgewidth=2)
# ax.text(x_monit + offset, y_monit - offset, "Critical location", color="white", fontsize=14,
#         ha="left", va="bottom", weight="bold")

# # Marker for Extraction
# ax.plot(x_ext, y_ext, marker="^", color="red", markersize=10, markeredgewidth=2)
# ax.text(x_ext + offset, y_ext + offset, "Extraction", color="white", fontsize=14,
#         ha="left", va="bottom", weight="bold")

# # Marker for Injection
# ax.plot(x_Inj, y_Inj, marker="v", color="red", markersize=10, markeredgewidth=2)
# ax.text(x_Inj + offset, y_Inj + offset, "Injection", color="white", fontsize=14,
#         ha="left", va="bottom", weight="bold")

# ax[:tick_params](axis="both", which="major", labelsize=14)
# display(fig)
# fig.savefig("Figure5a3d.pdf", dpi=600)
# PyPlot.close(fig)


fig, ax = PyPlot.subplots()
h_matrix = reshape(pres[3,:,:], n, n)
# Create coordinate vectors for axes (assuming grid from xmin to xmax, ymin to ymax)
xmin, xmax = -sidelength, sidelength  # replace with your actual range
ymin, ymax = -sidelength, sidelength  # replace with your actual range
x = range(xmin, xmax, length=n)
y = range(ymin, ymax, length=n)
# Plot filled contour with coordinates
contour = ax.contourf(x, y, h_matrix, cmap="viridis", levels=20, aspect="auto" )
ax.set_aspect("equal", adjustable="box")
ax.autoscale()
# Set axis limits explicitly
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
# ax.plot(x_monit, y_monit, marker="o", color="red", markersize=10, markeredgewidth=2)

cb = fig.colorbar(contour)
cb.ax.tick_params(labelsize=12)  # Adjust 12 to your desired font size
cb.set_label("Pressure (MPa)", fontsize=14)
ax[:tick_params](axis="both", which="major", labelsize=14)
display(fig)
fig.savefig("pressure3d.pdf", dpi=600)
PyPlot.close(fig)

