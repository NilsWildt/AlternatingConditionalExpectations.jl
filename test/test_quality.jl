using AlternatingConditionalExpectations
using Test

@testitem "Aqua quality checks" begin
    using Aqua
    Aqua.test_all(AlternatingConditionalExpectations)
end

@testitem "JET static analysis" begin
    using JET
    JET.test_package(AlternatingConditionalExpectations; target_defined_modules = true)
end
