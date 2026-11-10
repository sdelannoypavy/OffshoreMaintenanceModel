# setup.jl
#
# Loads the code of the experiments of paper.tex, in dependency order. Every script starts with
#   include(joinpath(@__DIR__, "..", "src", "setup.jl"))
# The environment variable GUROBI_THREADS sets the number of threads of Gurobi (the same for every
# method when computing times are compared).

include(joinpath(@__DIR__, "instances.jl"))        # instances of the full and simplified models
include(joinpath(@__DIR__, "model.jl"))            # common definitions: data, joint actions, costs, transitions, policies
include(joinpath(@__DIR__, "joint_actions.jl"))    # enumeration of the joint actions
include(joinpath(@__DIR__, "simulation.jl"))       # Monte Carlo simulation of a policy

# methods of paper.tex
include(joinpath(@__DIR__, "two_step_baseline.jl"))   # 2SB
include(joinpath(@__DIR__, "cflp.jl"))                # CFLP
include(joinpath(@__DIR__, "jflp_enum.jl"))           # JFLP (enum.), JFLP-Q
include(joinpath(@__DIR__, "jflp_cg.jl"))             # JFLP (CG), JFLP+cuts, JFLP-2
include(joinpath(@__DIR__, "anticipative.jl"))        # best anticipative policy (lower bound)

haskey(ENV, "GUROBI_THREADS") && Gurobi.GRBsetintparam(GRB_ENV, "Threads", parse(Int, ENV["GUROBI_THREADS"]))

"""
    make_policy(method, fd; H = 6, gamma = 1.0, cache_file = nothing, log_solve = info -> nothing)

Policy of a method of paper.tex, named as in the tables:
  "2SB", "CFLP", "JFLP-enum", "JFLP-CG", "JFLP-cuts", "JFLP-2", "JFLP-Q".
"""
function make_policy(method::String, fd::FluidData; H::Int = 6, gamma::Float64 = 1.0,
                     cache_file::Union{String,Nothing} = nothing, log_solve::Function = info -> nothing)
    kw = (H = H, cache_file = cache_file, log_solve = log_solve)
    method == "2SB"       && return make_policy_two_step_baseline(fd; gamma = gamma, log_solve = log_solve)
    method == "CFLP"      && return make_policy_cflp(fd; kw...)
    method == "JFLP-enum" && return make_policy_jflp_enum(fd; kw...)
    method == "JFLP-Q"    && return make_policy_jflp_enum(fd; quota_subsystem = true, kw...)
    method == "JFLP-CG"   && return make_policy_jflp_cg(fd; kw...)
    method == "JFLP-cuts" && return make_policy_jflp_cg(fd; cuts = true, kw...)
    method == "JFLP-2"    && return make_policy_jflp_cg(fd; pairs = same_pole_pairs(fd), kw...)
    error("unknown method $method")
end

const METHODS = ["2SB", "CFLP", "JFLP-enum", "JFLP-CG", "JFLP-cuts", "JFLP-2", "JFLP-Q"]
