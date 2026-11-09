# grouped_maintenance_share.jl
#
# Share of "grouped" maintenance events in a *_periods.csv file (run_experiment.jl /
# simulation.jl): a maintenance event (one component, maintained during one period, for one
# scenario) is grouped if its day window [start, start+d-1] overlaps -- shares at least one
# day, not merely the same 2-month period -- with another maintained component's window in
# that SAME period. Usage:
#
#   julia --project=. paper_experiment/grouped_maintenance_share.jl <instance> <periods_csv> [<periods_csv> ...]

using Printf

include(joinpath(@__DIR__, "instances.jl"))
include(joinpath(@__DIR__, "fluid_lp_cg.jl"))

function grouped_share(fd::FluidData, path::String)
    d = fd.d
    n_events = 0
    n_grouped = 0
    open(path) do io
        readline(io)   # header
        for line in eachline(io)
            f = split(line, ",")
            maint = f[7]
            isempty(maint) && continue
            ms = parse.(Int, split(maint, "-"))
            n_events += length(ms)
            length(ms) < 2 && continue
            starts = parse.(Int, split(f[8], "-"))
            ranges = [(starts[i], starts[i] + d[ms[i]] - 1) for i in eachindex(ms)]
            for i in eachindex(ms)
                a, b = ranges[i]
                for j in eachindex(ms)
                    i == j && continue
                    c, e = ranges[j]
                    if a <= e && c <= b
                        n_grouped += 1
                        break
                    end
                end
            end
        end
    end
    return n_grouped, n_events
end

instance_name = ARGS[1]
fd = build_fluid_data(Dict(i.name => i for i in instances)[instance_name])
for path in ARGS[2:end]
    n_grouped, n_events = grouped_share(fd, path)
    pct = n_events == 0 ? 0.0 : 100 * n_grouped / n_events
    @printf("%-70s  grouped=%d/%d  (%.1f%%)\n", path, n_grouped, n_events, pct)
end
