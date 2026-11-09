# plot_action.jl
#
# Generates a Gantt-style PDF plot of a maintenance action.
# Called from simulate_cg.jl whenever a non-trivial action is applied.
#
# Usage:
#   plot_and_save_action(col, d, C, T_per, label, output_dir)

using Plots
gr()

const COMPONENT_NAMES_PLOT = [
    "Converter", "Water circuit", "Fans", "Pumps",
    "Transformer", "Cooling", "Busbar"
]

const BAR_COLORS_PLOT = [
    :steelblue, :darkorange, :green3, :crimson,
    :mediumpurple, :saddlebrown, :teal
]

"""
    plot_and_save_action(col, d, C, T_per, label, output_dir)

Render a Gantt-style chart of the maintenance action `col` and save as PDF.

Arguments:
  col        : MaintenanceColumn (fields: plan.comps, plan.start_days, free_days)
  d          : nominal maintenance durations [working days] per component, length ≥ C
  C          : number of components
  T_per      : period length [calendar days]
  label      : string identifier used in the filename (e.g. "scen01_action0042")
  output_dir : destination folder (created if absent)

Layout:
  Y-axis rows 1..C = components 1..C (bottom to top)
  X-axis           = calendar days 1..T_per
  Free days        = yellow vertical bands across all component rows
"""
function plot_and_save_action(col, d::Vector{Int}, C::Int, T_per::Int,
                               label::String, output_dir::String)
    mkpath(output_dir)

    y_labels = [c <= length(COMPONENT_NAMES_PLOT) ? COMPONENT_NAMES_PLOT[c] : "Comp $c"
                for c in 1:C]

    plot_height = max(380, 55 * C)
    plt = plot(
        xlims         = (0.5, T_per + 0.5),
        ylims         = (0.5, C + 0.5),
        yticks        = (1:C, y_labels),
        xlabel        = "Day of period",
        title         = "",
        legend        = :outertop,
        legendcolumns = -1,      # all entries on one horizontal row
        size          = (1000, plot_height),
        grid          = :x,
        framestyle    = :box,
        left_margin   = 12Plots.mm,
        bottom_margin = 8Plots.mm,
        top_margin    = 8Plots.mm,
        right_margin  = 8Plots.mm,
    )

    # Yellow vertical bands for free days (across all component rows)
    first_free = true
    for τ in col.free_days
        plot!(plt,
            Shape([τ - 0.5, τ + 0.5, τ + 0.5, τ - 0.5],
                  [0.5, 0.5, C + 0.5, C + 0.5]),
            fillcolor = :gold, linecolor = :transparent, alpha = 0.35,
            label = first_free ? "Free days" : "")
        first_free = false
    end

    # Maintenance bars — one rectangle per maintained component
    for (i, c) in enumerate(col.plan.comps)
        s     = col.plan.start_days[i]
        e     = s + d[c] - 1
        bar_c = BAR_COLORS_PLOT[mod1(c, length(BAR_COLORS_PLOT))]

        plot!(plt,
            Shape([s - 0.42, e + 0.42, e + 0.42, s - 0.42],
                  [c - 0.33, c - 0.33, c + 0.33, c + 0.33]),
            fillcolor = bar_c, linecolor = bar_c, alpha = 0.80,
            label = "")
    end

    # Axis ticks: every 5 days + first and last
    major_ticks = sort(unique(vcat(1, collect(5:5:T_per), T_per)))
    plot!(plt, xticks = major_ticks)

    pdf_path = joinpath(output_dir, "action_$(label).pdf")
    savefig(plt, pdf_path)
    @printf("  [plot_action] Saved: %s\n", pdf_path)
    return pdf_path
end
