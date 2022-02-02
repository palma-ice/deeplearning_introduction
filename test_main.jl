using Flux, CairoMakie, PlutoUI, CUDA, Random, ProgressMeter

nx = 4
ny = 4
nm = 1000
range_param = 1

Random.seed!(1)
p = round.( range_param .* 2 .* (Random.rand(ny) .- 0.5), digits=2)
Random.seed!(1)
M = round.( range_param .* 2 .* (Random.rand(ny,nx) .- 0.5), digits=2)

function linearmap(x)
	Random.seed!(1)
	return M * x .+ p .+ randn(ny)
end

linearmap([4.91, 11.90, 39.32, 2.40])

Random.seed!(1)
X = 100 .* rand(nx, nm)
Y = mapslices(linearmap, X; dims=1)

layer1 = Dense(nx, ny)		# Create the layer
reshape(layer1.W, (nx,ny))	# Display values of initialisation


predict = Chain(layer1)

loss(x,y) = Flux.Losses.mse(predict(x), y)
loss(X[:,1], Y[:,1])


ps = params(predict)
# gs = Flux.gradient(ps) do
    # loss(x,y)
# end
gs(x,y) = Flux.gradient(() -> loss(x, y), ps)
gs[W]

