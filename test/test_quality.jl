using ACE
using Test

@testitem "Aqua quality checks" begin
    using Aqua
    Aqua.test_all(ACE)
end

@testitem "JET static analysis" begin
    using JET
    JET.test_package(ACE; target_defined_modules = true)
end
