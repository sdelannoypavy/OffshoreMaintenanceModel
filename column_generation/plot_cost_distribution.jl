using CSV, DataFrames, StatsPlots, Statistics, Printf, Distributions

cost_dir = joinpath(@__DIR__, "..", "cost_files")

df_cg   = CSV.read(joinpath(cost_dir, "cost_cg.csv"),               DataFrame, header=false)
df_dlp  = CSV.read(joinpath(cost_dir, "cost_direct_lp.csv"),        DataFrame, header=false)
df_enh  = CSV.read(joinpath(cost_dir, "cost_cg_enhanced.csv"),      DataFrame, header=false)
df_full = CSV.read(joinpath(cost_dir, "cost_cg_full_enhanced.csv"), DataFrame, header=false)

costs_cg   = df_cg[:,  2]
costs_dlp  = df_dlp[:, 2]
costs_enh  = df_enh[:, 2]
costs_full = df_full[:, 2]

cvar(x, α) = mean(x[x .>= quantile(x, 1 - α)])

function ci_mean(x; α=0.05)
    n      = length(x)
    t_crit = quantile(TDist(n - 1), 1 - α / 2)
    hw     = t_crit * std(x) / sqrt(n)
    return mean(x) - hw, mean(x) + hw
end

for (name, costs) in [("CG", costs_cg), ("LP relaxation", costs_dlp),
                       ("extended CG", costs_enh), ("full extended CG", costs_full)]
    lo, hi = ci_mean(costs)
    @printf("%-24s  n=%4d  mean=%7.2f  95%% CI=[%7.2f,%7.2f]  std=%7.2f  CVaR_95=%7.2f\n",
            name, length(costs), mean(costs), lo, hi, std(costs), cvar(costs, 0.05))
end

# ── Violin plot: CG / LP relaxation / full extended CG ───────────────────────
methods    = ["Relaxation", "Full LP", "Full LP with pairs"]
costs_list = [costs_dlp, costs_cg, costs_full]

p = plot(; ylabel="Total cost",
           xticks=(1:3, methods), xlims=(0.5, 3.5), ylims=(0, 500),
           legend=false)

for (i, costs) in enumerate(costs_list)
    boxplot!(p, [i], costs;
        label         = false,
        color         = i,
        fillalpha     = 0.5,
        linewidth     = 1.5,
        outliers      = false,
        whisker_range = 1.5)
end

savefig(p, joinpath(@__DIR__, "cost_distribution.pdf"))
display(p)
