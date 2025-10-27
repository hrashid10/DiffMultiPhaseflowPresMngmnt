import DPFEHM
import GaussianRandomFields
import Random
import BSON
import PyPlot
import Flux


@BSON.load "mytrained_model_Finetuned.bson" model

mutable struct Fluid_n
    vw::Float64
    vo::Float64
    swc::Float64
    sor::Float64
end

# Random.seed!(9) #for paper plots #case 1 (5a 5b)
Random.seed!(1794) #for paper plots #case 2 high pressure (HIghest) (5c 5d)
# Random.seed!(178) #for paper plots #case 3 high pressure (2nd) (5e 5f)


Qinj = 0.031688 # [m^3/s] (1 MMT water/yr)
n = 26
ns = [n, n]
steadyhead = 0e0
sidelength = 500
thickness  = 1.0
mins = [-sidelength, -sidelength] #meters
maxs = [sidelength, sidelength] #meters
num_eigenvectors = 200
sigma = 1.0
lambda = 50
cov = GaussianRandomFields.CovarianceFunction(2, GaussianRandomFields.Matern(lambda, 1; σ=sigma))
x_pts = range(mins[1], maxs[1]; length=ns[1])
y_pts = range(mins[2], maxs[2]; length=ns[2])
grf = GaussianRandomFields.GaussianRandomField(cov, GaussianRandomFields.KarhunenLoeve(num_eigenvectors), x_pts, y_pts)

pressure_target  = 0
learning_rate = 1e-4

coords, neighbors, areasoverlengths, volumes = DPFEHM.regulargrid2d(mins, maxs, ns, thickness)
dirichletnodes = Int[]
dirichleths = zeros(size(coords, 2))
specificstorage = fill(0.1, size(coords, 2))  
h0 = zeros(size(coords, 2))
fluid=Fluid_n(1.0, 1.0, 0.0, 0.0)
S0=zeros(size(coords, 2))

nt = 45;  dt =8*24*60*60;

# Calculate distance between extraction and injection wells
monitoring_well_node = 190
injection_extraction_nodes = [271, 487]

for i = 1:size(coords, 2)
    if abs(coords[1, i]) == sidelength || abs(coords[2, i]) == sidelength
        push!(dirichletnodes, i)
        dirichleths[i] = steadyhead
    end
end

function getQs(Qs::Vector, is::Vector)
    sum(Qs .* ((collect(1:size(coords, 2)) .== i) for i in is))
end

function solve_numerical(Qs, T)
    Qs = getQs(Qs, injection_extraction_nodes)
    everystep=false # output all the time steps
    Ts=exp.(reshape(T,size(Qs)))
    args=h0, S0, Ts, dirichleths,  dirichletnodes, Qs, volumes, areasoverlengths, fluid, dt, neighbors, nt, everystep
    h_gw_t, S= DPFEHM.solvetwophase(args...)
    push!(pressure_vals, h_gw_t[monitoring_well_node] - steadyhead)
    push!(extraction_rates, Qs[injection_extraction_nodes[1]])
    return h_gw_t[monitoring_well_node] - steadyhead
end

T = GaussianRandomFields.sample(grf)
Ts=exp.(reshape(T,size(coords,2)))
Tst=reshape(Ts,n,n,1,1)
everystep=false
Q1 = model(Tst)
# Q1=[0.0]
Qs = zeros(size(coords, 2))
@show Qs[injection_extraction_nodes[1]]=Q1[1] 
Qs[injection_extraction_nodes[2]]=Qinj
args=h0, S0, Ts, dirichleths,  dirichletnodes, Qs, volumes, areasoverlengths, fluid, dt, neighbors, nt, everystep
@time h_gw_t, S= DPFEHM.solvetwophase(args...)
@show h_gw_t[monitoring_well_node]


fig, ax = PyPlot.subplots(constrained_layout=true)
xmin, xmax = -sidelength, sidelength
ymin, ymax = -sidelength, sidelength

x = range(xmin, xmax, length=n)
y = range(ymin, ymax, length=n)
const mpl = PyPlot.matplotlib
# Plot using imshow with correct spatial coordinates using extent
img = ax.imshow(
    reshape(Ts, n, n),
    origin="lower",
    cmap="viridis",
    extent=[xmin, xmax, ymin, ymax],
    aspect="equal",  # or use "equal"
    interpolation="bicubic",
    norm=mpl.colors.LogNorm()
)

cb = fig.colorbar(img)
cb.ax.tick_params(labelsize=18)  # Adjust 12 to your desired font size
cb.set_label("Permeability (m²)", fontsize=18)

# Cell numbers to mark
monit = monitoring_well_node
ext = injection_extraction_nodes[1]
Inj = injection_extraction_nodes[2]

# Convert linear indices to (row, column)
row_monit = div(monit - 1, n) + 1
col_monit = mod(monit - 1, n) + 1

row_ext = div(ext - 1, n) + 1
col_ext = mod(ext - 1, n) + 1

row_Inj = div(Inj - 1, n) + 1
col_Inj = mod(Inj - 1, n) + 1

# Convert indices to coordinates
x_monit = x[col_monit]
y_monit = y[row_monit]

x_ext = x[col_ext]
y_ext = y[row_ext]

x_Inj = x[col_Inj]
y_Inj = y[row_Inj]

offset = sidelength * 0.05  # adjust as needed for spacing

# Marker for Monitoring Well
ax.plot(x_monit, y_monit, marker="o", color="red", markersize=14, markeredgewidth=2)
ax.text(x_monit + offset, y_monit - offset, "Critical location", color="white", fontsize=18,
        ha="left", va="bottom", weight="bold")

# Marker for Extraction
ax.plot(x_ext, y_ext, marker="^", color="red", markersize=14, markeredgewidth=2)
ax.text(x_ext + offset, y_ext + offset, "Extraction", color="white", fontsize=18,
        ha="left", va="bottom", weight="bold")

# Marker for Injection
ax.plot(x_Inj, y_Inj, marker="v", color="red", markersize=14, markeredgewidth=2)
ax.text(x_Inj - 2*offset, y_Inj + offset, "Injection", color="white", fontsize=18,
        ha="left", va="bottom", weight="bold")
ax.set_xlabel("x (m)", fontsize=18)
ax.set_ylabel("y (m)", fontsize=18)

ax[:tick_params](axis="both", which="major", labelsize=18)
display(fig)
fig.savefig("Figure5b.pdf",  bbox_inches="tight", pad_inches=0.02)
PyPlot.close(fig)


fig, ax = PyPlot.subplots()
h_matrix = reshape(h_gw_t, n, n)
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
ax.plot(x_monit, y_monit, marker="o", color="red", markersize=14, markeredgewidth=2)

cb = fig.colorbar(contour)
cb.ax.tick_params(labelsize=18)  # Adjust 12 to your desired font size
cb.set_label("Pressure (MPa)", fontsize=18)
ax[:tick_params](axis="both", which="major", labelsize=18)

ax.set_xlabel("x (m)", fontsize=18)
ax.set_ylabel("y (m)", fontsize=18)

display(fig)
fig.savefig("Figure5e.pdf",  bbox_inches="tight", pad_inches=0.02)
PyPlot.close(fig)

