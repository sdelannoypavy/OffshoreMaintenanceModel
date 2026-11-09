# plot_cost_distributions.jl
#
# Plots the distribution of the total simulated cost of several methods, side by side, from
# their *_costs.csv files (run_experiment.jl / simulation.jl, column `total_cost`): one
# coloured block per method, spanning the 25th-75th percentile, a line at the median, and
# whiskers to the 5th/95th percentile (StatsPlots' `boxplot` is not a dependency of this
# project, so the box is drawn by hand with Plots.jl's `Shape`).
#
# Usage:
#   julia --project=. paper_experiment/plot_cost_distributions.jl <label>=<file> [<label>=<file> ...] [-o out.pdf] [-t "title"]
#
# Example (the four methods of the paper's "Simplified model" table, one_pole_simple):
#   julia --project=. paper_experiment/plot_cost_distributions.jl \
#     "Baseline 1"=paper_experiment/results/final_results/one_pole_simple_20260927_091719_baseline1_costs.csv \
#     "Compact Fluid LP"=paper_experiment/results/final_results/one_pole_simple_20260927_124646_compact_costs.csv \
#     "Full Fluid LP"=paper_experiment/results/final_results/one_pole_simple_20260927_091719_fluid_cg_costs.csv \
#     "Lift-and-project"=paper_experiment/results/final_results/one_pole_simple_20260927_091719_lift_project_costs.csv \
#     -o paper_experiment/results/final_results/cost_distributions_one_pole_simple.pdf \
#     -t "Simplified model, one substation (1000 scenarios)"
#
# Every `<label>=<file>` pair becomes one block, in the order given. `<file>` is a
# *_costs.csv file as written by run_experiment.jl (header includes `total_cost`); any CSV
# with that column works. -o (default: cost_distributions.pdf, next to the first file) sets
# the output path -- both a .pdf and a .png are written. -t sets the plot title (default:
# none).

using Plots, Statistics
gr()

function read_total_cost(path::String)
    lines = readlines(path)
    header = split(lines[1], ",")
    col = findfirst(==("total_cost"), header)
    isnothing(col) && error("$path: no `total_cost` column (header: $(lines[1]))")
    return [parse(Float64, split(l, ",")[col]) for l in lines[2:end]]
end

pairs = String[]; out_path = ""; title_str = ""
i = 1
while i <= length(ARGS)
    a = ARGS[i]
    if a == "-o"
        global out_path = ARGS[i + 1]; global i += 2
    elseif a == "-t"
        global title_str = ARGS[i + 1]; global i += 2
    else
        push!(pairs, a); global i += 1
    end
end
isempty(pairs) && error("usage: plot_cost_distributions.jl <label>=<file> [<label>=<file> ...] [-o out.pdf] [-t title]")

labels = String[]; files = String[]
for pair in pairs
    eq = findfirst('=', pair)
    isnothing(eq) && error("expected <label>=<file>, got \"$pair\"")
    push!(labels, pair[1:eq-1]); push!(files, pair[eq+1:end])
end
isempty(out_path) && (out_path = joinpath(dirname(files[1]), "cost_distributions.pdf"))

palette = [:steelblue, :darkorange, :seagreen, :indianred, :mediumpurple, :goldenrod, :teal, :salmon]

p = plot(size = (max(500, 150 * length(labels) + 200), 420), legend = false,
        ylabel = "Total simulated cost", title = title_str,
        xticks = (1:length(labels), labels), xlims = (0.4, length(labels) + 0.6),
        grid = :y, framestyle = :box)

for (i, (label, file)) in enumerate(zip(labels, files))
    d = read_total_cost(file)
    q05, q25, q50, q75, q95 = quantile(d, [0.05, 0.25, 0.5, 0.75, 0.95])
    w = 0.32
    color = palette[mod1(i, length(palette))]
    plot!(p, [i, i], [q05, q25], color = :black, lw = 1.2)
    plot!(p, [i, i], [q75, q95], color = :black, lw = 1.2)
    plot!(p, [i - 0.12, i + 0.12], [q05, q05], color = :black, lw = 1.2)
    plot!(p, [i - 0.12, i + 0.12], [q95, q95], color = :black, lw = 1.2)
    plot!(p, Shape([i - w, i + w, i + w, i - w], [q25, q25, q75, q75]),
          color = color, fillalpha = 0.55, linecolor = :black, lw = 1.2)
    plot!(p, [i - w, i + w], [q50, q50], color = :black, lw = 2.2)
    println(rpad(label, max(20, maximum(length, labels))),
            "  n=", length(d), "  q05=", round(q05, digits = 1), "  q25=", round(q25, digits = 1),
            "  median=", round(q50, digits = 1), "  q75=", round(q75, digits = 1),
            "  q95=", round(q95, digits = 1))
end

savefig(p, out_path)
png_path = splitext(out_path)[1] * ".png"
savefig(p, png_path)
println("saved ", out_path, " and ", png_path)
