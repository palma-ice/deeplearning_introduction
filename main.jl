### A Pluto.jl notebook ###
# v0.17.7

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ d9ca9886-8410-11ec-0f6b-957729a9c708
using Flux, CairoMakie, PlutoUI, Random, LinearAlgebra

# ╔═╡ 4e2c385b-1d4a-4478-949b-4ba5d684ef84
html"<button onclick='present()'>present</button>" 	# Activating presentation mode.

# ╔═╡ 4cb9993c-424b-49ad-af9b-5261c4fd6fac
md"""
## Deep Learning $\subset$ Machine Learning $\subset$ AI
"""

# ╔═╡ ccc2a564-77cf-4b30-9d11-ba7e6b8c57b9
Show(MIME"image/jpeg"(), read("images/DLoverview.jpeg"))

# ╔═╡ 8a006f3c-ad88-4d21-8914-ebb057bfda12
md"""
# Why Did Deep Learning Become so Popular?

+ General approximator.
+ Scalable.
+ Does not plateau with increase of data input.
+ Better computation ressources: GPU and cloud computing.
+ Packages allowing fast implementation (e.g. TensorFlow, PyTorch, Theano, Keras...). However this leads to a large community of users but a comparatively small one when it comes to understand the underlying principles!

**Examples of Application:**

+ Data analysis.
+ Image processing.
+ Time series prediction.
+ Control and system identification (learning the right-hand side of an ODE or even PDE).
+ Anomaly detection.
+ Natural language processing (NLP). Most famous example: GPT-3.
"""

# ╔═╡ 008bb14c-3bfa-41f5-82b6-321ec4545cb1
Show(MIME"image/jpg"(), read("images/scalability.jpg"))

# ╔═╡ 822e024f-819f-4b36-8cdb-8ca69174cd95
md"""
# Introduction to Deep Learning

In this tutorial, you'll be walked through:
+ An affine regression with a shallow network.
+ The same regression on noisy data.
+ A nonlinear regression with a deeper network.
+ The common problems that can arise when using deep learning methods.
+ Some previews of more complex architectures.

For this purpose we will use Julia as programming language, Pluto as notebook interface and Flux as deep learning package.
"""

# ╔═╡ 76b7ff5f-d9c8-4b31-8100-0aeae0cfaee6
md"""
# Linear Regression with Shallow Network

Making an affine (or linear) regression on some data is a problem for which reliable analytical solutions exist. **An artificial neural network (ANN) is therefore quite a bad choice for solving this problem!** However, such a simple example allows to understand well what happens under the hood. 
"""

# ╔═╡ fdea54b1-06d8-4ea2-814b-c825b9dec7e1
md"""
### Importing Packages

+ Flux: a deep learning package that allows you to go pretty low-level, while being still user-freindly.
+ Makie: a plotting package with several backend. Here the Cairo one has been chosen.
+ PlutoUI: a package allowing the use of widgets.
+ Random: a package for generating random outputs.
+ LinearAlgebra: will provide us the matrix-norm functions.
"""

# ╔═╡ 5036a12a-1563-4431-8f61-6b2c19b602c0
md"""
# Linear Function as Ground Truth

Here we will construct some data that should be fitted by the ANN. To this end we create a random but reproduceable matrix M and bias vector p.
"""

# ╔═╡ f1cb6788-88f1-4e13-8201-27642d6debd9
@bind nx PlutoUI.Slider(2:2:20, show_value=true, default=4)

# ╔═╡ a9f133e5-9eec-497b-ab75-30593eaa8ba7
@bind ny PlutoUI.Slider(2:2:20, show_value=true, default=4)

# ╔═╡ 432b0747-e0c2-4a0a-bcbb-0ee7ffce6e79
@bind nm PlutoUI.Slider(100:100:2000, show_value=true, default=1000)

# ╔═╡ 214da52c-47fa-44fd-b735-baf2ab665b5e
input_range = 100

# ╔═╡ 4a74a7ad-bc6e-488a-91de-2b44df7971de
begin
	Random.seed!(1)
	M = round.( 2 .* (Random.rand(ny,nx) .- 0.5), digits=2)
end

# ╔═╡ c75ad903-cfd0-4696-b180-ec18e48f5b83
md"""
# Generating Data

Now we create some input data $X = [x_1, x_2, ..., x_{n_m}]$ and compute the corresponding output $Y = [y_1, y_2, ..., y_{n_m}]$ of the linear mapping:

```math
\begin{equation}
y_{i} = M x_{i}, \quad x_i \in \mathbf{R}^{n_x}, \quad y_{i} \in \mathbf{R}^{n_y}.
\end{equation}
```
"""

# ╔═╡ 7048e5ff-483b-4d39-846a-ecbfe64cef50
function linearmap(x)
	return M * x
end

# ╔═╡ 9cbebc46-2af8-4631-b481-762b53bafad6
linearmap([4.91, 11.90, 39.32, 2.40])

# ╔═╡ 94c7cc46-b46c-4bfd-b73a-3730605aeb15
begin
	Random.seed!(1)
	X = input_range .* rand(nx, nm)
	Y = mapslices(linearmap, X; dims=1)
end

# ╔═╡ 02d772e2-07ed-4b8e-bf01-d7c5472fbdbd
md"""
# Build a Dense Layer

Dense layers have exactly the same structure as the above defined deterministic linear mapping! 

```math
\begin{equation}
y = W x
\end{equation}
```

The question now is: can we recover the entries of $M$ by only using the data pair $\left\lbrace X,Y \right\rbrace$? Mathematically speaking: 

```math
\begin{equation}
\mathrm{minimise} \quad  \mathrm{dist}( W, M )
\end{equation}
```


We will now initialise such a layer with random parameters. In deep learning, *parameters* describe any value that can be optimised during the loss minimisation, e.g. $W_{11}$. In opposition, *hyperparameters* describe numerical values that will not be tuned at training time! For instance we can think of the dimension of the matrix W.
"""

# ╔═╡ 6679b679-7246-4750-b731-17125f38d52a
begin
	layer1 = Dense(nx, ny, bias=false)		# Create the layer
	predict = Chain(layer1)
	reshape(layer1.W, (nx,ny))				# Display values of initialisation
end

# ╔═╡ ba98b184-7608-4a94-bfb7-5a3ad8f8f615
LocalResource("images/ExampleNN.png", :width => 300)

# ╔═╡ a541839d-932d-4c2a-af80-9cc2b5f787a7
md"""
# Constructing the Loss

The distance between model and reality is quantified by a loss w.r.t. the output of the model $\hat{Y}$. In other word, if the predicted data is similar to the ground truth, we consider the model to be suited. Our goal can now be formulated rigorously:

```math
\begin{equation}
\underset{W}{\mathrm{minimise}} \quad  L(\hat{Y}, Y)
\end{equation}
```

As an intuitive error measure, we choose the mean square error:

```math
\begin{eqnarray}
l(\hat{y}, y) &=& \dfrac{1}{2}(\hat{y} - y)^{\mathrm{T}}(\hat{y} - y) \\
L(\hat{Y}, Y) &=& \dfrac{1}{n_m} \underset{\hat{y}, y}{\Sigma} l(\hat{y}, y).
\end{eqnarray}
```
"""

# ╔═╡ 373f5103-9adc-426c-bbcd-d19a511463e6
loss(x,y) = Flux.Losses.mse(predict(x), y)

# ╔═╡ 06fbb7f6-10cd-4ada-bacf-2e44e5947f81
loss(X[:,1], Y[:,1])

# ╔═╡ f2d6ad9c-1348-4955-a390-ea23e998c734
md"""
# Gradient Descent

As we can see here, this value is very high and the model is thus rubbish for now. However, we can change the parameters to reduce the loss! This can be easily done with the gradient descent technique:

```math
\begin{equation}
W^{(i+1)} = W^{(i)} - \alpha \nabla_{W} l(\hat{y}, y),
\end{equation}
```

with $\alpha \in (0,1)$ the so-called *learning rate*. Computing the gradient can be easily done by the use of the chain rule. In our case, this yields:

```math
\begin{eqnarray}
\nabla_{W} l(\hat{y}, y) &=& \dfrac{\partial l}{\partial \hat{y}}\dfrac{\partial \hat{y}}{\partial W} \\
&=& (\hat{y}-y) \otimes x.
\end{eqnarray}
```

This gets quite tedious for large systems... but luckily, it is handled by any deep learning package.
"""

# ╔═╡ d5e60ca5-cfce-40db-b284-d9a278f6d73e
begin
	α = 1e-4
	ps = params(predict)
	gs(x,y) = Flux.gradient(() -> loss(x,y), ps)
	grad1 = gs(X[:,1], Y[:,1])
	dW = grad1.grads[grad1.params[1]]
	predict.layers[1].W .-= α .* dW
end

# ╔═╡ 6254c67c-f4ce-4286-adfa-ab13c1555178
loss(X[:,1], Y[:,1])

# ╔═╡ 4157d2b4-7946-45ce-b77e-ebdc63744594
md"""
Hurray! The cost decreased and if we repeat this several times, we might get a near-zero loss!
"""

# ╔═╡ 5f51257d-0296-42c5-8908-a25afbd63bf8
md"""
# Split the Data

Before going to the full training procedure, we perform a common step called *data-splitting*. While a fraction $f_1$ is used for training, a fraction $f_2$ is used to control the generalisation error during the training. Finally a fraction $f_3$ is kept aside to evaluate the generalisation performance on data that was never evaluated by the ANN. Commonly, $f_2 = f_3$ is chosen. Typical values of $f$ are:
+ $f_1 \in [0.5, 0.9]$
+ $f_2, f_3 \in [0.05, 0.25]$
"""

# ╔═╡ 8f5c2f4a-03a9-472f-a0ee-31200d631454
function split_data(f1, f2, nm, X, Y)
	ntrain = Int(round(f1*nm, digits=0))
	ndev = Int(round((f1+f2)*nm, digits=0))
	Xtrain, Ytrain = X[:, 1:ntrain], Y[:, 1:ntrain]
	Xdev, Ydev = X[:, ntrain:ndev], Y[:, ntrain:ndev]
	Xtest, Ytest = X[:, ndev:nm], Y[:, ndev:nm]
	return ntrain, Xtrain, Ytrain, Xdev, Ydev, Xtest, Ytest
end

# ╔═╡ 56f95a4e-49b8-4b81-9cd6-6d632090a068
begin
	f1, f2, = 0.7, 0.15
	ntrain, Xtrain, Ytrain, Xdev, Ydev, Xtest, Ytest = split_data(f1, f2, nm, X, Y)
end

# ╔═╡ b2a47c06-14b6-4136-ab7f-3adafdc84836
size(Xtrain), size(Xdev), size(Xtest)

# ╔═╡ e80866d2-fa32-40db-871f-fe43734b90a8
md"""
# Build Batches of Training Data

Another data-splitting that is commonly performed is to seperate the training data in batches of size $n_b$. This allows to update the parameters on a small number of experiments, thus avoiding some drawbacks:
+ If $n_b = 1$, we get our previous update method. This is noisy and not well-suited in the vicinity of the minimum.
+ If $n_b = n_m$, updating takes a long time and we miss some noisiness to leave local minima.
"""

# ╔═╡ 0f8f171a-1e70-4ee9-a242-4933c7588bd1
@bind batch_size PlutoUI.Slider(10:10:200, show_value=true, default=100)

# ╔═╡ 0e659769-229c-47ea-8db8-9ab59e750162
train_loader = Flux.Data.DataLoader( (data=Xtrain, label=Ytrain), 
									  batchsize=batch_size, shuffle=true)

# ╔═╡ ab43a786-4aaa-4d25-8f8f-d77e414f10f9
md"""
# Train the Network

Now we want to stack everything we have seen until now to obtain our linear regression!
"""

# ╔═╡ 312d7fe3-216b-4b3d-98bb-498138a9aba4
@bind n_epochs PlutoUI.Slider(2:2:50, show_value=true, default=20)

# ╔═╡ 122a1c6f-5fb5-4045-a297-b57c1f4a15d6
function train_network(ne::Int, optimiser, loader, Xd, Yd, p, l::Function)
	tloss, dloss = [], []
	for epoch in 1:ne
		for (x,y) in loader
			g = Flux.gradient(p) do
				l(x,y)
			end
			Flux.update!(optimiser, p, g)
			append!(tloss, l(x,y))
			append!(dloss, l(Xd, Yd))
		end
	end
	return tloss, dloss
end

# ╔═╡ 60b84bf0-b566-43f7-858b-27a0bc88a82c
begin
	opt = Descent(α)
	@time trainloss, devloss = train_network(n_epochs, opt, train_loader, Xdev, Ydev, ps, loss)
end

# ╔═╡ fcac5bcb-5d04-4b86-8426-c03ef117b45c
md"""
From worker 2:      0.005 seconds (26.61 k allocations: 12.665 MiB)
"""

# ╔═╡ 614f8685-75d9-4b98-8096-65d0e133d2b6
md"""
# Results Analysis
"""

# ╔═╡ 3f55e36d-314b-4c47-a352-b860035fd353
function plot_loss(ntrain, batch_size, trainloss, devloss)
	scale_epoch = ntrain/batch_size
	fig = Figure(resolution = (800, 400))
	ax = CairoMakie.Axis(fig[1,1], xlabel="epochs", ylabel="loss", 
		      xminorticks=IntervalsBetween(5), xminorgridvisible=true)
	lines!(ax, (1:length(trainloss)) ./  scale_epoch, Float32.(trainloss))
	lines!(ax, (1:length(devloss)) ./ scale_epoch , Float32.(devloss))
	fig
end

# ╔═╡ a769d445-ec6c-4dca-95c5-2eb4a47566a2
@time plot_loss(ntrain, batch_size, trainloss, devloss)

# ╔═╡ a0ef458b-04ad-4612-ac8f-956ff86d377a
predict.layers[1].W, M

# ╔═╡ 242499eb-036e-457c-8f26-58bb6a75e844
md"""
# Noisy Data

As real-world data is commonly affected by noise, we add a white noise to the data:

```math
\begin{equation}
y_{i} = M x_{i} + \nu, \quad \nu \sim \mathcal{N}(0.5, 1), \quad \nu \in \mathbf{R}^{n_y}.
\end{equation}
```
"""

# ╔═╡ c648fc8f-c32a-4321-8714-f50903a238f3
function noisy_linearmap(x)
	return M * x .+ (0.5 .+ randn(ny))
end

# ╔═╡ 970566aa-afa9-4e8d-b4d7-f2f1bcbab84b
begin
	Random.seed!(1)
	Xn = input_range .* rand(nx, nm)
	Yn = mapslices(noisy_linearmap, X; dims=1)
end

# ╔═╡ 264cd113-41a9-4b1e-bfef-314bf042114c
ntrainn, Xntrain, Yntrain, Xndev, Yndev, Xntest, Yntest = split_data(f1, f2, nm, Xn, Yn)

# ╔═╡ 9506744e-1b89-4fa9-ad0a-55c1cb0a1105
train_loadern = Flux.Data.DataLoader( (data=Xntrain, label=Yntrain), 
									   batchsize=batch_size, shuffle=true)

# ╔═╡ 0e39448d-5880-4f4e-b994-85b185bce0d3
md"""
# Build a Blank Model & Train it
"""

# ╔═╡ f89822ef-4611-468e-9969-e0eca14a3d58
begin
	layer1n = Dense(nx, ny)
	predictn = Chain(layer1n)
	psn = params(predictn)
	lossn(x,y) = Flux.Losses.mse(predictn(x), y)
	trainlossn, devlossn = train_network(n_epochs, opt, train_loadern, Xndev, Yndev, psn, lossn)
	plot_loss(ntrainn, batch_size, trainlossn, devlossn)
end

# ╔═╡ 16910f68-be7b-4600-ad2c-4516b9eb61f2
md"""
# Compare Results with Clean Data
"""

# ╔═╡ 6d141430-ee39-4c84-a0a8-5b8becccfc3c
predictn.layers[1].W, M

# ╔═╡ 2e8120f9-2c3b-4f7a-9e12-8233fba3578d
md"""
To compare the error in the clean and in the noisy case, we take the 2-norm of the matrix difference:
"""

# ╔═╡ 47e703f6-e2a1-44c9-a3a9-4b87f68c00c5
norm(predict.layers[1].W-M, 2), norm(predictn.layers[1].W-M, 2)

# ╔═╡ d4777405-ba8f-4593-acd4-678bd5bda3e2
md"""
What if we don't know the ground truth matrix $M$? We kept the test data especially for this!
"""

# ╔═╡ 22f0b115-bca8-412a-97c0-0d102c4b646d
loss(Xtest, Ytest), lossn(Xntest, Yntest)

# ╔═╡ 8dccc75a-8afd-47d4-b25e-0f2542ec9b1a
md"""
# Generate Nonlinear Data

First, let us generate some data based on a nonlinear function:

$f(x) = \sum_{i} x_i^{e_i},$

with $e_i \in [0,1]$ some randomly sampled but fixed exponents.
"""

# ╔═╡ 87b6fdf8-6fa4-404c-b074-f5e940b6e6fb
begin
	Random.seed!(1)
	exponents = rand(nx)
end

# ╔═╡ ed438d19-7926-45e1-83d3-2f4d20455f7b
function nonlinearmap(x)
	# return sum(x .^ exponents)
	return cos(2*x[1]) + cos(x[2]) .^ 2 + x[3] + sin(x[4])
end

# ╔═╡ 86fe5883-91d6-42be-b544-90fde2cbad9b
begin
	Random.seed!(1)
	Xnl = input_range .* rand(nx, nm)
	Ynl = mapslices(nonlinearmap, Xnl; dims=1)
end

# ╔═╡ a6926a79-252c-41da-a9e3-1f2f9ab9bf21
begin
	ntrainnl, Xnltrain, Ynltrain, Xnldev, Ynldev, Xnltest, Ynltest = split_data(f1, f2, nm, Xnl, Ynl)
	train_loadernl = Flux.Data.DataLoader( (data=Xnltrain, label=Ynltrain), 
									   batchsize=batch_size, shuffle=true)
end

# ╔═╡ e6540e9e-86c1-4425-940d-f606f76e6aff
md"""
# Towards a More Elaborate NN

Fitting nonlinear data can be achieved by applying a so-called *activation* on the output of a dense layer:

$z = \Phi(Wx+b).$

Common choices for $\Phi$ are:

+ Sigmoid or tanh.
+ Softmax.
+ Rectified Linear Unit (ReLU) or Leaky ReLU.
+ Or custom!

Notice that these activations serve the purpose of avoiding vanishing and exploding gradients, as well as produce outputs with suited ranges.
"""

# ╔═╡ 9abf3b1f-e62d-436d-8b10-a92f708bb81c
Show(MIME"image/png"(), read("images/Activation.png"))

# ╔═╡ 84053e01-8c47-4dd7-b1f4-fb2c43a957f2
md"""
# Stacking Layers

Good news: we can still perform a gradient descent (or anything similar) while accepting a higher degree of complexity.

Now with a single layer, we only allow a low degree of nonlinearity. To tackle this problem, there is a straight forward solution: let's stack some layers! This results in nothing more than having a matrix $W^{(i)}$, a bias $b^{(i)}$ and an activation $\Phi^{(i)}$ for each layer $i \in {1, ..., L}$.

Thanks to computational graphs, computing the gradient of such stacked layers can be easily done!
"""

# ╔═╡ 0e9a77d8-4108-41b6-8c0e-2cf5b92c7baf
begin
	layer1nl, layer2nl = Dense(nx, nx, leakyrelu), Dense(nx, nx, leakyrelu)
	layer3nl, layer4nl = Dense(nx, nx, leakyrelu), Dense(nx, nx, leakyrelu)
	layer5nl = Dense(nx, 1)
	predictnl = Chain(layer1nl, layer2nl, layer3nl, layer4nl, layer5nl)
end

# ╔═╡ 84d593c1-a8a6-419b-b100-09beef602f72
Show(MIME"image/png"(), read("images/deepnet.png"))

# ╔═╡ bdb756b7-ecb8-4c9e-8f25-f0a993522b78
md"""
# Training the Network

Here we use a different optimiser to obtain a better performance w.r.t. training process.
"""

# ╔═╡ 9387b001-a175-4aa9-918d-8bd83b465348
@bind nln_epochs PlutoUI.Slider(20:20:2000, show_value=true, default=20)

# ╔═╡ 3c8a043b-1602-44f8-ba0e-c6f3a40c0dde
begin
	psnl = params(predictnl)
	optnl = RADAM(1e-4, (0.9, 0.8))
	lossnl(x,y) = Flux.Losses.mse(predictnl(x), y)
	lossnl(Xnl[:,1], Ynl[1])
end

# ╔═╡ ab47a8fc-b36a-4b02-891c-af773ab752e1
trainlossnl, devlossnl = train_network(nln_epochs, optnl, train_loadernl, 
										   Xnldev, Ynldev, psnl, lossnl)

# ╔═╡ 31fc72c5-3791-4c6c-b331-bac1a6d7357f
lossnl(Xnl[:,1], Ynl[1])

# ╔═╡ 7bb7e89e-7ef3-48e3-a8cd-a5721a61a474
md"""
# Analysing the Results
"""

# ╔═╡ ace85058-c14f-44db-88d8-51273b8fb0f7
plot_loss(ntrainnl, batch_size, trainlossnl, devlossnl)

# ╔═╡ ea485131-cd71-4646-9abe-6ddb90eb2c28
predictnl(Xnltest[:,1:5])

# ╔═╡ 6e11e766-74b6-4d31-a0b1-0689745d48e1
Ynltest[1:5]

# ╔═╡ 5e117cb5-c360-47fd-be60-41bfcb076550
md"""
# What Could Possibly Go Wrong?

### Optimisation-Related Problems
+ Zero-initialisation of W.
+ Exploding or vanishing gradient. Both particularly likely in deep networks!
+ Loss is not adapted to your actual requirements.
+ Local minima. Less likely with increasing complexity.
+ Noisiness avoids to find the minimum.

### Structure-Related Problems
+ The structure is too simple to perform well, even on the training data.
+ Basic structure might be good, but sometimes you need to scale it up! Example: ConvLSTM network for pedestrian motion prediction with more than a million parameters.
+ Structure has too many parameters to be trained in reasonable time.

### Data-Related Problems
+ Data too sparse or not well pre-processed. Always compare the number of parameter you want to train and the number of data points.
+ Lack of generality = perform well on data but not on real applications. Underlying problem: overfitting or data provided by different distributions.

### Building a Neural Network is an Iterative Process!
1. Think about the data you are dealing with!
1. Start with easiest architecture.
1. Check that it is able to fit your training data. If not, go back to 2.
1. Check that it is able to fit test data. If not, obtain more data or apply regularisation techniques.
1. Improve your performance by tuning hyperparameters.
"""

# ╔═╡ 31a1aa20-57f5-4463-8d61-82b10c045799
md"""
# Why Do ANN Perform so Well?

+ Unconstrained optimisation is a straightforward task.
+ Structure simple enough to compute gradient analytically.
+ Structure **scalable**.
+ High dimensional optimisation: local minima unlikely!
"""

# ╔═╡ fe35dc87-3c0b-4568-974c-05611fbf5835
Show(MIME"image/png"(), read("images/LocalMin.png"))

# ╔═╡ 862737ee-e7ca-45f4-ba93-5571393b6571
md"""
#### Why Deepen Instead of Broaden?

+ The complexity follows an exponential behavior with depth. Only linear with width. Thus less parameters for same complexity.
+ Allows to modify dimensions to achieve some nice properties.
"""

# ╔═╡ d1c7aa96-989f-4ef6-b99f-3dd591336ae1
md"""
# Further Concepts

+ MSE not always the best! For example: cross-entropy.
+ More elaborate optimisers perform better then gradient descent!
+ Any more complex architecture of ANN relies on the same basic principles!
"""

# ╔═╡ 2d5dc63c-437e-47a5-a50f-9fb58de7fc32
md"""
### Important Structures You Might Hear Of

+ Recurrent Neural Networks (RNN). Particularly suited to tackle the problem of gradient vanishing in time series analysis. The most famous of them: LSTM and GRU.
+ Convolutional Neural Networks (CNN). Particularly suited for image processing because it tackles the curse of dimensionality and has nice invariance properties.
+ Convolutional Recurent Neural Networks (CRNN). Particularly well suited for video analysis/prediction.
+ Generative Adversarial Networks (GANs). Two networks, a classifier and a generator, compete against each other. After a while, the generator is able to trick the classifier... and the human eye (deep fakes).
+ Auto-encoders.
"""

# ╔═╡ 43409cd7-1c41-4c49-b0db-10de210a29f9
Show(MIME"image/png"(), read("images/Autoencoder.png"))

# ╔═╡ 9100898e-14c2-4dc6-b806-c70a1da99c20
md"""
# Recommendation for Learning More

+ Andrew Ng course(s) on Youtube or Coursera.
+ Ian Goodfellow: *Deep Learning*
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
CairoMakie = "~0.7.2"
Flux = "~0.12.9"
PlutoUI = "~0.7.32"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.0"
manifest_format = "2.0"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "6f1d9bc1c08f9f4a8fa92e3ea3cb50153a1b40d4"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.1.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e81c509d2c8e49592413bfb0bb3b08150056c79d"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.1"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "1bdcc02836402d104a46f7843b6e6730b1948264"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "4.0.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Automa]]
deps = ["Printf", "ScanByte", "TranscodingStreams"]
git-tree-sha1 = "d50976f217489ce799e366d9561d56a98a30d7fe"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "0.8.2"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random", "Test"]
git-tree-sha1 = "a598ecb0d717092b5539dbbe890c98bac842b072"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.2.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "215a9aa4a1f23fbd05b92769fdd62559488d70e9"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.1"

[[deps.CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CompilerSupportLibraries_jll", "ExprTools", "GPUArrays", "GPUCompiler", "LLVM", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "SpecialFunctions", "TimerOutputs"]
git-tree-sha1 = "7405a853ebba81936827459bfe30608bbce7371e"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "3.8.0"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "d0b3f8b4ad16cb0a2988c6788646a5e6a17b6b1b"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.0.5"

[[deps.CairoMakie]]
deps = ["Base64", "Cairo", "Colors", "FFTW", "FileIO", "FreeType", "GeometryBasics", "LinearAlgebra", "Makie", "SHA", "StaticArrays"]
git-tree-sha1 = "90fe6622efbb627e7c962e9bd6f5c4228680b7ca"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.7.2"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.ChainRules]]
deps = ["ChainRulesCore", "Compat", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "Statistics"]
git-tree-sha1 = "849d4cb467ea3ecbbd3efe68dacd36f9429b543c"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.26.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "f9982ef575e19b0e5c7a98c6e75ee496c0f73a93"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.12.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON", "Test"]
git-tree-sha1 = "61c5334f33d91e570e1d0c3eb5465835242582c4"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "6b6f04f93710c71550ec7e16b650c1b9a612d0b6"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.16.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "3f1f500312161f1ae067abe07d13b40f78f32e07"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.8"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "44c37b4636bc54afac5c574d2d02b625349d6582"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.41.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "84083a5136b6abf426174a58325ffd159dd6d94f"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.9.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "24d26ca2197c158304ab2329af074fbe14c988e4"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.45"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.EllipsisNotation]]
deps = ["ArrayInterface"]
git-tree-sha1 = "d7ab55febfd0907b285fbf8dc0c73c0825d9d6aa"
uuid = "da5c29d0-fa7d-589e-88eb-ea29b0a81949"
version = "1.3.0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[deps.ExprTools]]
git-tree-sha1 = "56559bbef6ca5ea0c0818fa5c90320398a6fbf8d"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.8"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "463cb335fa22c4ebacfd1faba5fde14edb80d96c"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.5"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "67551df041955cc6ee2ed098718c8fcd7fc7aebe"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.12.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "8756f9935b7ccc9064c6eef0bff0ad643df733a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.7"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Flux]]
deps = ["AbstractTrees", "Adapt", "ArrayInterface", "CUDA", "CodecZlib", "Colors", "DelimitedFiles", "Functors", "Juno", "LinearAlgebra", "MacroTools", "NNlib", "NNlibCUDA", "Pkg", "Printf", "Random", "Reexport", "SHA", "SparseArrays", "Statistics", "StatsBase", "Test", "ZipFile", "Zygote"]
git-tree-sha1 = "983271b47332fd3d9488d6f2d724570290971794"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.12.9"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "1bd6fc0c344fc0cbee1f42f8d2e7ec8253dda2d2"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.25"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "cabd77ab6a6fdff49bfd24af2ebe76e6e018a2b4"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.0.0"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics", "StaticArrays"]
git-tree-sha1 = "770050893e7bc8a34915b4b9298604a3236de834"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.9.5"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Functors]]
git-tree-sha1 = "e4768c3b7f597d5a352afa09874d16e3c3f6ead2"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.2.7"

[[deps.GPUArrays]]
deps = ["Adapt", "LLVM", "LinearAlgebra", "Printf", "Random", "Serialization", "Statistics"]
git-tree-sha1 = "cf91e6e9213b9190dc0511d6fff862a86652a94a"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "8.2.1"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "abd824e1f2ecd18d33811629c781441e94a24e81"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.13.11"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "1c5a84319923bea76fa145d49e93aa4394c73fc2"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.1"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "70938436e2720e6cb8a7f2ca9f1bbdbf40d7f5d0"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.6.4"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "006127162a51f0effbdfaab5ac0c83f8eb7ea8f3"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.4"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "9a5c62f231e5bba35695a20988fc7cd6de7eeb5a"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.3"

[[deps.ImageIO]]
deps = ["FileIO", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "816fc866edd8307a6e79a575e6585bfab8cef27f"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.0"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "87f7662e03a649cffa2e05bf19c303e168732d3e"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.2+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "b15fc0a95c564ca2e0a7ae12c1f095ca848ceb31"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.5"

[[deps.IntervalSets]]
deps = ["Dates", "EllipsisNotation", "Statistics"]
git-tree-sha1 = "3cc368af3f110a767ac786560045dceddfc16758"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.5.3"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[deps.Juno]]
deps = ["Base64", "Logging", "Media", "Profile"]
git-tree-sha1 = "07cb43290a840908a771552911a6274bc6c072c7"
uuid = "e5e0dc1b-0480-54bc-9374-aad01c23163d"
version = "0.8.4"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "591e8dc09ad18386189610acafb970032c519707"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.3"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "f8dcd7adfda0dddaf944e62476d823164cccc217"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "4.7.1"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "62115afed394c016c2d3096c5b85c407b48be96b"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.13+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "e5718a00af0ab9756305a0392832c8952c7426c1"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.6"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "5455aef09b40e5020e1520f551fa3135040d4ed0"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2021.1.1+2"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Makie]]
deps = ["Animations", "Base64", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "Distributions", "DocStringExtensions", "FFMPEG", "FileIO", "FixedPointNumbers", "Formatting", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageIO", "IntervalSets", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MakieCore", "Markdown", "Match", "MathTeXEngine", "Observables", "OffsetArrays", "Packing", "PlotUtils", "PolygonOps", "Printf", "Random", "RelocatableFolders", "Serialization", "Showoff", "SignedDistanceFields", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "UnicodeFun"]
git-tree-sha1 = "0aafd5121c6e1b6a83bd3bb341da45f058225a9b"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.16.3"

[[deps.MakieCore]]
deps = ["Observables"]
git-tree-sha1 = "c5fb1bfac781db766f9e4aef96adc19a729bc9b2"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.2.1"

[[deps.MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.Match]]
git-tree-sha1 = "1d9bc5c1a6e7ee24effb93f175c9342f9154d97f"
uuid = "7eb4fadd-790c-5f42-8a69-bfa0b872bfbf"
version = "1.2.0"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "Test"]
git-tree-sha1 = "70e733037bbf02d691e78f95171a1fa08cdc6332"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.2.1"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Media]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "75a54abd10709c01f1b86b84ec225d26e840ed58"
uuid = "e89f7d12-3494-54d1-8411-f7d8b9ae1f27"
version = "0.5.0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "b34e3bc3ca7c94914418637cb10cc4d1d80d877d"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.3"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NNlib]]
deps = ["Adapt", "ChainRulesCore", "Compat", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "0b9f48ee4a793ae1d2c766093e9f45f405de6231"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.8.0"

[[deps.NNlibCUDA]]
deps = ["CUDA", "LinearAlgebra", "NNlib", "Random", "Statistics"]
git-tree-sha1 = "adb43860f6371f42fb42d1d792c5d0d2f4cbc716"
uuid = "a00861dc-f156-4864-bf3c-e6376f28a68d"
version = "0.2.0"

[[deps.NaNMath]]
git-tree-sha1 = "b086b7ea07f8e38cf122f5016af580881ac914fe"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.7"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore"]
git-tree-sha1 = "18efc06f6ec36a8b801b23f076e3c6ac7c3bf153"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.Observables]]
git-tree-sha1 = "fe29afdef3d0c4a8286128d4e45cc50621b1e43d"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.4.0"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "043017e0bdeff61cfbb7afeb558ab29536bbb5ed"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.8"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "923319661e9a22712f24596ce81c54fc0366f304"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.1+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "648107615c15d4e09f7eca16307bc821c1f718d8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.13+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "ee26b350276c51697c9c2d88a072b339f9f03d73"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.5"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "6d105d40e30b635cfed9d52ec29cf456e27d38f8"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.3.12"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "1155f6f937fa2b94104162f01fa400e192e4272f"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.4.2"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "03a7a85b76381a3d04c7a1656039197e70eda03d"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.11"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9bc1871464b12ed19297fbc56c4fb4ba84988b0d"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.47.0+0"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "0b5cfbb704034b5b4c1869e36634438a047df065"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.1"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "a7a7e1a88853564e551e4eba8650f8c38df79b37"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.1.1"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "6f1b25e8ea06279b5689263cc538f51331d7ca17"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.1.3"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "ae6145ca68947569058866e443df69587acc1806"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.32"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "2cf929d64681236a2e074ffafb8d568733d2e6af"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "afadeba63d90ff223a6a48d2009434ecee2ec9e8"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.1"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Random123]]
deps = ["Libdl", "Random", "RandomNumbers"]
git-tree-sha1 = "0e8b146557ad1c6deb1367655e052276690e71a3"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.4.2"

[[deps.RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "01d341f502250e81f6fec0afe662aa861392a3aa"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.2"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.SIMD]]
git-tree-sha1 = "39e3df417a0dd0c4e1f89891a281f82f5373ea3b"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.4.0"

[[deps.ScanByte]]
deps = ["Libdl", "SIMD"]
git-tree-sha1 = "9cc2955f2a254b18be655a4ee70bc4031b2b189e"
uuid = "7b38b023-a4d7-4c5e-8d43-3f3097f304eb"
version = "0.3.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SignedDistanceFields]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "d263a08ec505853a5ff1c1ebde2070419e3f28e9"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.0"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "8fb59825be681d451c246a795117f317ecbcaa28"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.2"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e6bf188613555c78062842777b116905a9f9dd49"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.0"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "b4912cd034cdf968e06ca5f943bb54b17b97793a"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.5.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "a635a9333989a094bddc9f940c04c549cd66afcf"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.3.4"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
git-tree-sha1 = "d88665adc9bcf45903013af0982e2fd05ae3d0a6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "51383f2d367eb3b444c961d485c565e4c0cf4ba0"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.14"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f35e1879a71cca95f4826a14cdbf0b9e253ed918"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.15"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "d21f2c564b21a202f4677c0fba5b5ee431058544"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.4"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "bb1064c9a84c52e277f1096cf41434b675cd368b"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "991d34bbff0d9125d93ba15887d6594e8e84b305"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.5.3"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "97e999be94a7147d0609d0b9fc9feca4bf24d76b"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.15"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "3593e69e469d2111389a9bd06bac1f3d730ac6de"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.9.4"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "IRTools", "InteractiveUtils", "LinearAlgebra", "MacroTools", "NaNMath", "Random", "Requires", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "88a4d79f4e389456d5a90d79d53d1738860ef0a5"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.34"

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[deps.isoband_jll]]
deps = ["Libdl", "Pkg"]
git-tree-sha1 = "a1ac99674715995a536bbce674b068ec1b7d893d"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.2+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "78736dab31ae7a53540a6b752efc61f77b304c5b"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.8.6+1"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"
"""

# ╔═╡ Cell order:
# ╟─4e2c385b-1d4a-4478-949b-4ba5d684ef84
# ╟─4cb9993c-424b-49ad-af9b-5261c4fd6fac
# ╟─ccc2a564-77cf-4b30-9d11-ba7e6b8c57b9
# ╟─8a006f3c-ad88-4d21-8914-ebb057bfda12
# ╟─008bb14c-3bfa-41f5-82b6-321ec4545cb1
# ╟─822e024f-819f-4b36-8cdb-8ca69174cd95
# ╟─76b7ff5f-d9c8-4b31-8100-0aeae0cfaee6
# ╟─fdea54b1-06d8-4ea2-814b-c825b9dec7e1
# ╠═d9ca9886-8410-11ec-0f6b-957729a9c708
# ╟─5036a12a-1563-4431-8f61-6b2c19b602c0
# ╠═f1cb6788-88f1-4e13-8201-27642d6debd9
# ╠═a9f133e5-9eec-497b-ab75-30593eaa8ba7
# ╠═432b0747-e0c2-4a0a-bcbb-0ee7ffce6e79
# ╠═214da52c-47fa-44fd-b735-baf2ab665b5e
# ╠═4a74a7ad-bc6e-488a-91de-2b44df7971de
# ╟─c75ad903-cfd0-4696-b180-ec18e48f5b83
# ╠═7048e5ff-483b-4d39-846a-ecbfe64cef50
# ╠═9cbebc46-2af8-4631-b481-762b53bafad6
# ╠═94c7cc46-b46c-4bfd-b73a-3730605aeb15
# ╟─02d772e2-07ed-4b8e-bf01-d7c5472fbdbd
# ╠═6679b679-7246-4750-b731-17125f38d52a
# ╟─ba98b184-7608-4a94-bfb7-5a3ad8f8f615
# ╟─a541839d-932d-4c2a-af80-9cc2b5f787a7
# ╠═373f5103-9adc-426c-bbcd-d19a511463e6
# ╠═06fbb7f6-10cd-4ada-bacf-2e44e5947f81
# ╟─f2d6ad9c-1348-4955-a390-ea23e998c734
# ╠═d5e60ca5-cfce-40db-b284-d9a278f6d73e
# ╠═6254c67c-f4ce-4286-adfa-ab13c1555178
# ╟─4157d2b4-7946-45ce-b77e-ebdc63744594
# ╟─5f51257d-0296-42c5-8908-a25afbd63bf8
# ╠═8f5c2f4a-03a9-472f-a0ee-31200d631454
# ╠═56f95a4e-49b8-4b81-9cd6-6d632090a068
# ╠═b2a47c06-14b6-4136-ab7f-3adafdc84836
# ╟─e80866d2-fa32-40db-871f-fe43734b90a8
# ╠═0f8f171a-1e70-4ee9-a242-4933c7588bd1
# ╠═0e659769-229c-47ea-8db8-9ab59e750162
# ╟─ab43a786-4aaa-4d25-8f8f-d77e414f10f9
# ╠═312d7fe3-216b-4b3d-98bb-498138a9aba4
# ╠═122a1c6f-5fb5-4045-a297-b57c1f4a15d6
# ╠═60b84bf0-b566-43f7-858b-27a0bc88a82c
# ╟─fcac5bcb-5d04-4b86-8426-c03ef117b45c
# ╟─614f8685-75d9-4b98-8096-65d0e133d2b6
# ╠═3f55e36d-314b-4c47-a352-b860035fd353
# ╠═a769d445-ec6c-4dca-95c5-2eb4a47566a2
# ╠═a0ef458b-04ad-4612-ac8f-956ff86d377a
# ╟─242499eb-036e-457c-8f26-58bb6a75e844
# ╠═c648fc8f-c32a-4321-8714-f50903a238f3
# ╠═970566aa-afa9-4e8d-b4d7-f2f1bcbab84b
# ╠═264cd113-41a9-4b1e-bfef-314bf042114c
# ╠═9506744e-1b89-4fa9-ad0a-55c1cb0a1105
# ╟─0e39448d-5880-4f4e-b994-85b185bce0d3
# ╠═f89822ef-4611-468e-9969-e0eca14a3d58
# ╟─16910f68-be7b-4600-ad2c-4516b9eb61f2
# ╠═6d141430-ee39-4c84-a0a8-5b8becccfc3c
# ╟─2e8120f9-2c3b-4f7a-9e12-8233fba3578d
# ╠═47e703f6-e2a1-44c9-a3a9-4b87f68c00c5
# ╟─d4777405-ba8f-4593-acd4-678bd5bda3e2
# ╠═22f0b115-bca8-412a-97c0-0d102c4b646d
# ╟─8dccc75a-8afd-47d4-b25e-0f2542ec9b1a
# ╠═87b6fdf8-6fa4-404c-b074-f5e940b6e6fb
# ╠═ed438d19-7926-45e1-83d3-2f4d20455f7b
# ╠═86fe5883-91d6-42be-b544-90fde2cbad9b
# ╠═a6926a79-252c-41da-a9e3-1f2f9ab9bf21
# ╟─e6540e9e-86c1-4425-940d-f606f76e6aff
# ╟─9abf3b1f-e62d-436d-8b10-a92f708bb81c
# ╟─84053e01-8c47-4dd7-b1f4-fb2c43a957f2
# ╠═0e9a77d8-4108-41b6-8c0e-2cf5b92c7baf
# ╟─84d593c1-a8a6-419b-b100-09beef602f72
# ╟─bdb756b7-ecb8-4c9e-8f25-f0a993522b78
# ╠═9387b001-a175-4aa9-918d-8bd83b465348
# ╠═3c8a043b-1602-44f8-ba0e-c6f3a40c0dde
# ╠═ab47a8fc-b36a-4b02-891c-af773ab752e1
# ╠═31fc72c5-3791-4c6c-b331-bac1a6d7357f
# ╟─7bb7e89e-7ef3-48e3-a8cd-a5721a61a474
# ╠═ace85058-c14f-44db-88d8-51273b8fb0f7
# ╠═ea485131-cd71-4646-9abe-6ddb90eb2c28
# ╠═6e11e766-74b6-4d31-a0b1-0689745d48e1
# ╟─5e117cb5-c360-47fd-be60-41bfcb076550
# ╟─31a1aa20-57f5-4463-8d61-82b10c045799
# ╟─fe35dc87-3c0b-4568-974c-05611fbf5835
# ╟─862737ee-e7ca-45f4-ba93-5571393b6571
# ╟─d1c7aa96-989f-4ef6-b99f-3dd591336ae1
# ╟─2d5dc63c-437e-47a5-a50f-9fb58de7fc32
# ╟─43409cd7-1c41-4c49-b0db-10de210a29f9
# ╟─9100898e-14c2-4dc6-b806-c70a1da99c20
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
