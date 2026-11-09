# time9.jl
#
# Computing time of the first decision (all components new, full quota, first period of the year, horizon 6) of the
# Full fluid LP + column generation on the instance of interactions9.jl with S substations, with the log of every
# iteration of the column generation (master LP time, pricing time, new columns). Usage:
#
#   [DISTINCT=1] MTBF_ALL=20 QUOTA_STEP=15 GUROBI_THREADS=3 julia --project=. paper_experiment/time9.jl <S>

S_t = parse(Int, ARGS[1])
outdir = mktempdir()
src = readlines(joinpath(@__DIR__, "interactions9.jl"))
stop = findfirst(l -> startswith(l, "fdS ="), src)
src = filter(l -> !startswith(l, "S = parse"), src[1:stop-1])
include_string(Main, join(src, "\n"))

fd = build_fluid_data(make_inst9(S_t))
println("S = $S_t, M = $(fd.M), DISTINCT = $DISTINCT, threads = $(get(ENV, "GUROBI_THREADS", "default"))"); flush(stdout)
col, info = solve_fluid_cg_paper(fd, ones(Int, fd.M), copy(fd.K0), 1, 6; verbose = true)
@printf("TOTAL %.1f s  pricing %.1f s  master %.1f s  iterations %d  converged %s  columns %d  LP %.6f\n",
        info.time, info.t_pricing, info.t_master, info.iterations, info.converged, info.n_columns, info.lp_value)
