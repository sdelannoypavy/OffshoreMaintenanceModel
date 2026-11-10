# anticipative_ratios.jl
#
# For every method of the "Simplified model" table, ratio of its cost to the cost of the best
# anticipative policy (anticipative.jl), scenario by scenario, on the scenarios simulated for
# that method. Prints the mean and the (sample) variance of the ratio. Usage:
#
#   julia --project=. paper_experiment/anticipative_ratios.jl
#
# Reads paper_experiment/results/final_results/<instance>_<...>_<method>_costs.csv and
# <instance>_anticipative_costs.csv.

using Printf, Statistics

const DIR = joinpath(@__DIR__, "results", "final_results")
const METHODS = [("baseline1", "2 steps"), ("compact", "Compact Fluid LP"),
                 ("fluid_cg", "Full Fluid LP"), ("lift_project", "Lift-and-project")]

function read_costs(path::String, key::String, value::String)
    lines = readlines(path); header = split(lines[1], ",")
    ik = findfirst(==(key), header); iv = findfirst(==(value), header)
    return Dict(parse(Int, split(l, ",")[ik]) => parse(Float64, split(l, ",")[iv]) for l in lines[2:end])
end

for instance in ("one_pole_simple", "two_poles_simple")
    antic = read_costs(joinpath(DIR, "$(instance)_anticipative_costs.csv"), "scenario", "anticipative_cost")
    println("\n== $instance  ($(length(antic)) anticipative scenarios, mean anticipative cost = ",
            round(mean(values(antic)); digits = 2), ")")
    @printf("  %-18s %6s %12s %14s %12s\n", "method", "n", "mean ratio", "variance ratio", "min ratio")
    for (method, label) in METHODS
        file = only(filter(f -> startswith(f, instance * "_") && endswith(f, "_$(method)_costs.csv"), readdir(DIR)))
        costs = read_costs(joinpath(DIR, file), "scenario", "total_cost")
        scen = sort([s for s in keys(costs) if haskey(antic, s)])
        r = [costs[s] / antic[s] for s in scen]
        @assert minimum(r) >= 1 - 1e-9 "$method: a policy beats the anticipative optimum on some scenario"
        @printf("  %-18s %6d %12.4f %14.4f %12.4f\n", label, length(r), mean(r), var(r), minimum(r))
    end
end
