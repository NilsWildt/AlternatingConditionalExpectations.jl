using DrWatson
@quickactivate "ACE"
using LinearAlgebra
using Statistics
using StatsBase
using Random
using FixedSizeArrays

include(srcdir("ACE.jl"))

using .ACE
using Plots
include(srcdir("Smoother.jl"))
using .Smoothers



println("Testing ACE Implementation")
println("=" ^ 50)

# Test 1: Generate bivariate data (Float64)
println("\n1. Generating bivariate test data (Float64)...")
N = 800
X, Y = ACE.generate_bivariate_data(Float64, N, 2.0, 2.0, 42)
println("   Generated $(N) samples")
println("   X shape: $(size(X))")
println("   Y shape: $(size(Y))")
println("   X type: $(typeof(X))")
println("   Y type: $(typeof(Y))")

# Test 2: Create smoother
println("\n2. Creating smoother...")
window_size = 5
smoother = Smoothers.LAS(window_size)
println("   Using LLSSb smoother with window size: $window_size")

# Test 3: Create ACE simulation
println("\n3. Setting up ACE simulation...")
myace = ACE.ACEsim(X, Y, smoother; 
                   errorbound=1e-12, 
                   itermax_inner=20, 
                   itermax_outer=50)
println("   Error bound: $(myace.errorbound)")
println("   Max inner iterations: $(myace.itermax_inner)")
println("   Max outer iterations: $(myace.itermax_outer)")

# Test 4: Run ACE algorithm
println("\n4. Running ACE algorithm...")
result = ACE.run(myace)
println("   ✓ ACE completed successfully!")
println("   Total iterations: $(result.itercount)")
println("   Computation time: $(round(result.t, digits=3)) seconds")
println("   Final ε²: $(round(result.ρ, digits=4))")
println("   Original correlation: $(round(result.r_orig[1], digits=4))")
println("   Transformed correlation: $(round(result.r²[1], digits=4))")

# Test 5: Test with multivariate data
println("\n5. Testing with multivariate data...")
N = 150
m = 3  # number of predictors
X_multi = randn(N, m)
Y_multi = vec(exp.(sum(X_multi.^2, dims=2) .+ 0.1 * randn(N)))

smoother_multi = Smoothers.LLSSb(11)
myace_multi = ACE.ACEsim(X_multi, Y_multi, smoother_multi; 
                         errorbound=1e-3, 
                         itermax_inner=15, 
                         itermax_outer=30)

result_multi = ACE.run(myace_multi)
println("   ✓ Multivariate ACE completed!")
println("   Number of predictors: $m")
println("   Total iterations: $(result_multi.itercount)")
println("   Final ε²: $(round(result_multi.ρ, digits=4))")

# Test 6: Test different smoothers
println("\n6. Testing different smoothers...")
smoothers_to_test = [
    ("LAS", Smoothers.LAS(11)),
    ("LASb", Smoothers.LASb(11)),
    ("LLSS", Smoothers.LLSS(11)),
    ("LLSSb", Smoothers.LLSSb(11))
]

X_test, Y_test = ACE.generate_bivariate_data(100, 1.0, 0.5, 123)
for (name, smoother) in smoothers_to_test
    myace_test = ACE.ACEsim(X_test, Y_test, smoother; 
                            errorbound=1e-3, 
                            itermax_inner=10, 
                            itermax_outer=20)
    result_test = ACE.run(myace_test)
    println("   $name: ε² = $(round(result_test.ρ, digits=4)), iterations = $(result_test.itercount)")
end

# Test 7: Test plotting (if desired)
println("\n7. Testing visualization with CairoMakie...")
try
    fig = ACE.plot_ace_results(result; full=true)
    # Uncomment the following line to save the figure
    # save("ace_results.png", fig)
    println("   ✓ Plotting successful (figure created)")
catch e
    println("   Note: Plotting skipped or failed: $e")
end

# Test 8: Test with Float32
println("\n8. Testing with Float32 precision...")
X32, Y32 = ACE.generate_bivariate_data(Float32, 100, 1.0f0, 0.5f0, 456)
smoother32 = Smoothers.LLSSb(11)
myace32 = ACE.ACEsim(X32, Y32, smoother32; 
                     errorbound=1e-3, 
                     itermax_inner=10, 
                     itermax_outer=20)
result32 = ACE.run(myace32)
println("   ✓ Float32 ACE completed!")
println("   Final ε²: $(round(result32.ρ, digits=4))")
println("   Type check: ε² type = $(typeof(result32.ρ))")

println("\n" * "=" ^ 50)
println("All tests completed successfully!")
println("=" ^ 50)