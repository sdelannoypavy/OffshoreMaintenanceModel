# analysis.jl
#
# Numbers of the tables and figures of Section "Numerical experiments" of paper.tex, from the result
# files of scripts/simulate.jl and scripts/anticipative.jl. Usage (from the root of the repository):
#
#   julia --project=. paper_experiment/scripts/analysis.jl <instance> [resultsdir]
#
# For every method with results on the instance, over the scenarios simulated by every such method:
#   - mean, standard deviation and CVaR_0.95 of the total cost of a scenario,
#   - percentage of grouped maintenances: maintenances that share at least one day with the
#     maintenance of another component in the same period,
#   - mean computing time of a new decision, and its split between the construction of the models
#     (Build), the pricing MILPs (MILP) and the solves of the LPs (LP),
#   - quantiles 5/25/50/75/95 of the ratio of the cost of a scenario to its anticipative cost, over the
#     scenarios whose anticipative cost is positive (box plots of the figures).
# The rows are printed in the format of the tables of paper.tex.

include(joinpath(@__DIR__, "..", "src", "setup.jl"))

instance = ARGS[1]
resultsdir = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "..", "results")
fd = build_fluid_data(INSTANCES[instance])
T_total = fd.inst.n_years * N_PERIODS_PER_YEAR

"Rows of every csv file of `resultsdir` whose name matches `pattern` (header removed), split on commas."
function read_rows(pattern::Regex)
    rows = Vector{Vector{SubString{String}}}()
    for f in sort(readdir(resultsdir))
        occursin(pattern, f) || continue
        for l in readlines(joinpath(resultsdir, f))[2:end]
            push!(rows, split(l, ","))
        end
    end
    return rows
end

"Total cost of every complete scenario of a method, and its periods (columns of PERIODS_HEADER, src/simulation.jl)."
function scenario_costs(method::String)
    periods = Dict{Int,Vector{Vector{SubString{String}}}}()
    for r in read_rows(Regex("^" * instance * "_" * method * raw"_\d+_periods\.csv$"))
        push!(get!(periods, parse(Int, r[3]), []), r)
    end
    filter!(kv -> length(kv[2]) == T_total, periods)
    costs = Dict(s => sum(parse(Float64, r[10]) + parse(Float64, r[11]) for r in rs) for (s, rs) in periods)
    return costs, periods
end

"Numbers of grouped maintenances and of maintenances in the periods `rows`."
function grouped_maintenances(rows)
    n_grouped = 0; n_maint = 0
    for r in rows
        isempty(r[7]) && continue
        ms = parse.(Int, split(r[7], "-")); starts = parse.(Int, split(r[8], "-"))
        n_maint += length(ms)
        windows = [(starts[i], starts[i] + fd.d[ms[i]] - 1) for i in eachindex(ms)]
        for i in eachindex(ms)
            any(j != i && windows[i][1] <= windows[j][2] && windows[j][1] <= windows[i][2] for j in eachindex(ms)) && (n_grouped += 1)
        end
    end
    return n_grouped, n_maint
end

"Conditional value at risk: mean of the worst (1 − α) fraction of the costs."
cvar(costs, α = 0.95) = mean(sort(costs; rev = true)[1:max(1, ceil(Int, (1 - α) * length(costs)))])


methods = [m for m in METHODS if !isempty(scenario_costs(m)[1])]
isempty(methods) && error("no result for $instance in $resultsdir")
data = Dict(m => scenario_costs(m) for m in methods)
common = sort(collect(intersect((Set(keys(data[m][1])) for m in methods)...)))
@printf("%s: %d scenarios simulated by every method (%s)\n\n", instance, length(common), join(methods, ", "))

anticipative = Dict(parse(Int, r[1]) => parse(Float64, r[2]) for r in read_rows(Regex("^" * instance * raw"_anticipative_\d+\.csv$")))
ratio_scen = [s for s in common if get(anticipative, s, 0.0) > 0]

println("Method & Mean & Std & CVaR_0.95 & Grouped (%) & Time (s) & Build (%) & MILP (%) & LP (%) \\\\")
for m in methods
    costs = [data[m][1][s] for s in common]
    n_grouped, n_maint = grouped_maintenances(vcat((data[m][2][s] for s in common)...))
    solves = read_rows(Regex("^" * instance * "_" * m * raw"_\d+_solves\.csv$"))
    t = [parse(Float64, r[3]) for r in solves]; t_milp = [parse(Float64, r[4]) for r in solves]; t_lp = [parse(Float64, r[5]) for r in solves]
    T = sum(t; init = 0.0)
    split_cols = (m == "2SB" || T == 0) ? "-- & -- & --" :
        @sprintf("%.0f & %s & %.0f", 100 * (T - sum(t_milp) - sum(t_lp)) / T,
                 sum(t_milp) > 0 ? @sprintf("%.0f", 100 * sum(t_milp) / T) : "--", 100 * sum(t_lp) / T)
    @printf("%s & %.2f & %.2f & %.2f & %.1f & %.2f & %s \\\\\n", m, mean(costs), std(costs), cvar(costs),
            n_maint == 0 ? 0.0 : 100 * n_grouped / n_maint, isempty(t) ? NaN : mean(t), split_cols)
end

if !isempty(ratio_scen)
    @printf("\nRatio to the anticipative cost, %d scenarios (box plots: q05, q25, median, q75, q95; mean, std)\n", length(ratio_scen))
    for m in methods
        r = [data[m][1][s] / anticipative[s] for s in ratio_scen]
        @printf("%-10s \\costbox{}{}{%.3f}{%.3f}{%.3f}{%.3f}{%.3f}   mean %.3f  std %.3f\n", m,
                quantile(r, 0.05), quantile(r, 0.25), quantile(r, 0.5), quantile(r, 0.75), quantile(r, 0.95),
                mean(r), std(r))
    end
end
