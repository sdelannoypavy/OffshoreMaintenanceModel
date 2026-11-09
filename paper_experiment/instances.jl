# instances.jl
#
# Instances of the numerical experiments of the paper (Section 5).
# Three instances: one, two and three poles (K = 1, 2, 3).
# Each pole is a copy of the simple pole built in create_simple_pole.jl (7 components).
# On top of the simple pole data, every component has:
#   - a technician requirement (fraction of the total workforce R used per maintenance),
#   - a replacement cost.
# Each pole has its own quota of penalty-free maintenance days and its own daily penalties.
# The penalties of an instance are the sum of the penalties of each pole, incurred
# when the capacity of the pole is reduced.
#
# NOTE: technician shares, replacement costs (in M) and MTBF are the data of the paper.
# The quota of free days and the daily penalty are fixed placeholders, to be replaced by
# the real data (penalties must be in the same unit as the replacement costs): quota = 2
# free days/year, penalty = 1 per unit of capacity loss and per day, comparable in scale to
# the replacement costs (0.125 to 10).

include(joinpath(@__DIR__, "..", "create_simple_pole.jl"))
# provides: C, n, d, P_evac, P_daily, P_period, Capacity, ...

const T_PERIOD = 60          # days in a two-month period
const N_PERIODS_PER_YEAR = 6
const MARGIN = 0             # no maintenance may start during the last MARGIN days of the period
const LEAD_MARGIN = 15       # lead time L of paper.tex: it takes L days after a maintenance is
                              # planned before it can actually start (e.g. ordering parts).
# The start days are τ_start ∈ {L + 1, ..., 60} as in paper.tex: the first L days of the period are
# the lead time, and the earliest admissible start day of any component is FIRST_START_DAY = L + 1.
# With L = 15, a period of 60 days is then split exactly into 4 blocks of 15 days (the lead time and
# the 3 slots of the start-day grid 16, 31, 46 when QUOTA_STEP = 15), so that every part of a period
# on which the capacity loss of a pole is constant lasts a multiple of 15 days.
# The environment variable FIRST_START_DAY overrides it (e.g. FIRST_START_DAY=15 reproduces the former
# model, in which the repairs could start on day L).
const FIRST_START_DAY = parse(Int, get(ENV, "FIRST_START_DAY", string(LEAD_MARGIN + 1)))

struct Pole
    n::Vector{Int}                       # number of degradation states of each component
    d::Vector{Int}                       # repair duration (working days) of each component
    P_daily::Vector{Matrix{Float64}}     # daily degradation transition matrix of each component
    P_period::Vector{Matrix{Float64}}    # transition matrix over one period (60 days) if not maintained
    P_evac::Matrix{Float64}              # P_evac[c, x]: capacity (%) when component c is in state x
    technician_share::Vector{Float64}    # share of the workforce R needed by each maintenance
    replacement_cost::Vector{Float64}    # cost of replacing each component
    free_quota::Int                      # annual quota of penalty-free maintenance days
    daily_penalty::Vector{Float64}       # penalty per day and per unit of capacity loss, for each day of the period
end

struct Instance
    name::String
    n_poles::Int
    poles::Vector{Pole}
    R::Float64                           # total workforce available each day, shared by all poles (1.0 = 100%)
    T_period::Int
    margin::Int
    n_years::Int
    simplified::Bool                     # true for the *_simple instances: no within-period
                                          # degradation (a component that is not maintained
                                          # stays in its entering state for the whole period;
                                          # it only transitions, via P_period, between periods)
end

# Data of the paper (table of the components). Components:
# converter, water circuit, fans, pumps, transformers, converter cooling, busbar (output).
# MTBF are those of create_simple_pole.jl (transformers: 2222).
# Technician shares and repair durations: every component needs half the workforce (50%)
# and 15 workdays to repair, except the converter and the transformer, which need the
# whole workforce (100%) -- the converter's repair additionally takes 30 workdays (twice
# as long as every other component, including the transformer).
const TECHNICIAN_SHARE = [1.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5]   # share of the total number of technicians
const REPAIR_DURATION  = [30, 15, 15, 15, 15, 15, 15]          # workdays, overrides create_simple_pole.jl's d
const REPLACEMENT_COST = [5.0, 0.25, 0.125, 0.25, 10.0, 0.5, 0.25]   # in M

const FREE_QUOTA = 30                         # placeholder value (free days per year, per pole)
# Common mobilization cost (vessel, crew): paid once per period as soon as at least one component
# of the region is maintained, whatever the number of components and of poles maintained. It
# couples the poles by rewarding the grouping of their maintenances in the same period. Default 0
# (no mobilization cost), environment variable MOBILIZATION_COST to override.
const MOBILIZATION_COST = parse(Float64, get(ENV, "MOBILIZATION_COST", "0.0"))
const DAILY_PENALTY = fill(1.0, T_PERIOD)     # placeholder value (constant over the period)

# The remaining quota k, the free days used in a period κ and the start days of the repairs
# are tracked in steps of QUOTA_STEP days. With QUOTA_STEP = 1 (default), k, κ ∈ [0:K0] and
# every start day in {FIRST_START_DAY, ..., T − d_m + 1} is allowed: this is exactly the model of
# paper.tex. QUOTA_STEP = 15 (environment variable, e.g. `QUOTA_STEP=15 julia ...`) gives the
# former discretised model (k, κ ∈ {0, 15, 30}, start days on the grid 16, 31, 46), whose
# (k,κ) part of the state/action space is much smaller. qrange(top) is the resulting set of
# admissible values up to `top` (both k and top are always themselves multiples of
# QUOTA_STEP by construction, so this also correctly enumerates the admissible κ <= k for a
# given k).
const QUOTA_STEP = parse(Int, get(ENV, "QUOTA_STEP", "1"))
qrange(top::Int) = 0:QUOTA_STEP:top

# Data of the *_simple instances ONLY (the original instances keep the paper's TECHNICIAN_SHARE
# and MTBFs above): 0.3 instead of 0.5 for the half-workforce components, so that up to three of
# them (3 x 0.3 = 0.9 <= R) can be maintained simultaneously; and lower MTBFs (in years, same
# order: converter, water circuit, fans, pumps, transformer, cooling, busbar) so that several
# components are simultaneously worth maintaining (the paper's MTBFs make the converter,
# transformer and busbar essentially never fail over the horizon).
const TECHNICIAN_SHARE_SIMPLE = [1.0, 0.3, 0.3, 0.3, 1.0, 0.3, 0.3]
const MTBF_SIMPLE = [10.0, 4.0, 4.0, 4.0, 12.0, 6.0, 15.0]

"Daily transition matrices of the *_simple instances: same constructors as build_MC_utils.jl, with MTBF_SIMPLE."
function daily_matrices_simple()
    p_pump_s = 9.55e-5 * 43 / MTBF_SIMPLE[4]     # 9.55e-5 gives MTBF 43 (build_MC_utils.jl)
    p_trans_s = 1 / (365 * MTBF_SIMPLE[5])
    P_pump_s = [1 - 2*p_pump_s*(1 - p_pump_s)  0                 0
                2*p_pump_s*(1 - p_pump_s)      1 - p_pump_s      0
                0                              p_pump_s          1]
    P_trans_s = [1 - 2*p_trans_s*(1 - p_trans_s)  0                 0
                 2*p_trans_s*(1 - p_trans_s)      1 - p_trans_s     0
                 0                                p_trans_s         1]
    return [build_P_converter(MTBF_SIMPLE[1], n[1]), build_P_linear_daily(n[2], MTBF_SIMPLE[2]),
            build_P_fan(MTBF_SIMPLE[3]), P_pump_s, P_trans_s,
            build_P_linear_daily(n[6], MTBF_SIMPLE[6]), build_P_linear_daily(n[7], MTBF_SIMPLE[7])]
end

"""
    build_simple_pole(; free_quota=FREE_QUOTA, simplified=false)

Copy of the pole of create_simple_pole.jl with the data of the paper (technician shares,
replacement costs), completed with the (possibly overridden) quota of free days and the
placeholder daily penalty. `simplified = true` uses `TECHNICIAN_SHARE_SIMPLE` and `MTBF_SIMPLE`
instead of the paper's technician shares and MTBFs.
"""
function build_simple_pole(; free_quota::Int = FREE_QUOTA, simplified::Bool = false)
    # MTBF_FROM_SIMPLE=1 (environment variable): the original instances use MTBF_SIMPLE too, with
    # the rest of their data (quota, technician shares) unchanged
    simple_mtbf = simplified || get(ENV, "MTBF_FROM_SIMPLE", "0") == "1"
    Pd = simple_mtbf ? daily_matrices_simple() : deepcopy(P_daily)
    Pp = simple_mtbf ? [Pc^T_PERIOD for Pc in Pd] : deepcopy(P_period)
    share = simplified ? TECHNICIAN_SHARE_SIMPLE : TECHNICIAN_SHARE
    return Pole(copy(n), copy(REPAIR_DURATION), Pd, Pp, copy(P_evac),
                copy(share), copy(REPLACEMENT_COST), free_quota, copy(DAILY_PENALTY))
end

"""
    build_instance(n_poles; n_years=20, free_quota=FREE_QUOTA, name_suffix="")

Instance with `n_poles` poles, each being a copy of the simple pole. `free_quota = 0`
gives the simplified problem of `implemented_formulations.tex` (Section "Simplified
problem: no free-day quota"): `qrange(0) = 0:QUOTA_STEP:0 = {0}` collapses every pole's
(k, κ) to the single value 0, so no free day can ever be declared and `pole_penalty`'s
`free_day` branch is never taken -- the joint cost becomes exactly one unit of penalty per
day of ongoing maintenance per pole (since DAILY_PENALTY is uniformly 1), i.e. the sum,
over poles, of the number of days that pole is under maintenance. No change to the LP/MILP
code of fluid_lp_cg.jl, compact_lp.jl, fluid_cg_light.jl, fluid_cg_order2.jl or
lift_project.jl is needed to run this simplified problem: every method already runs on it
unchanged (this instance just makes their (k, κ) bookkeeping trivial), and it can be
selected by name exactly like the original instances.
"""
function build_instance(n_poles::Int; n_years::Int=20, free_quota::Int = FREE_QUOTA,
                        name_suffix::String = "", simplified::Bool = false)
    poles = [build_simple_pole(; free_quota = free_quota, simplified = simplified) for _ in 1:n_poles]
    names = Dict(1 => "one_pole", 2 => "two_poles", 3 => "three_poles")
    return Instance(get(names, n_poles, "$(n_poles)_poles") * name_suffix, n_poles, poles,
                    1.0, T_PERIOD, MARGIN, n_years, simplified)
end

instance_one_pole    = build_instance(1)
instance_two_poles   = build_instance(2)
instance_three_poles = build_instance(3)

# Simplified problem (Section "Simplified problem: no free-day quota" of
# implemented_formulations.tex): same poles/components/technician shares/repair
# durations/replacement costs, no free-day quota (free_quota = 0). Cost = Σ_m c_m(x_m, a_m)
# (failure penalties + replacement costs, unchanged) + Σ_p (number of days pole p is under
# maintenance) -- exactly the decomposable-MDP formalism of Section 2 of the paper, with no
# (k, κ) wrinkle. `simplified = true` additionally removes within-period degradation (Section
# "Simplified problem: no within-period transition" of implemented_formulations.tex): a
# component that is not maintained keeps its entering state for the whole period (no more
# day-by-day P_daily sampling in simulation.jl), and only transitions between periods, via
# P_period, exactly as the fluid LP itself already assumed. failure_cost_simplified
# (fluid_lp_cg.jl) computes δ_m(x) to match: the cost of staying in state x for the whole
# period, instead of failure_cost's day-by-day expectation.
instance_one_pole_simple    = build_instance(1; free_quota = 0, name_suffix = "_simple", simplified = true)
instance_two_poles_simple   = build_instance(2; free_quota = 0, name_suffix = "_simple", simplified = true)
instance_three_poles_simple = build_instance(3; free_quota = 0, name_suffix = "_simple", simplified = true)

instances = [instance_one_pole, instance_two_poles, instance_three_poles,
             instance_one_pole_simple, instance_two_poles_simple, instance_three_poles_simple]

"""
    pole_penalty(pole, state, ongoing_maintenance, free_day, τ)

Penalty of one pole on day τ of a period, due to the capacity reduction.
`state[c]` is the degradation state of component c, `ongoing_maintenance[c]` is true if
c is under maintenance on day τ, and `free_day` is true if τ is a penalty-free day.
"""
function pole_penalty(pole::Pole, state, ongoing_maintenance, free_day::Bool, τ::Int)
    C_pole = length(pole.n)
    if any(ongoing_maintenance)
        capacity_loss = 100.0                # the pole is shut down during maintenance
        return free_day ? 0.0 : pole.daily_penalty[τ] * capacity_loss / 100
    end
    capacity = minimum(pole.P_evac[c, state[c]] for c in 1:C_pole)
    return pole.daily_penalty[τ] * (100.0 - capacity) / 100    # failures reduce capacity, never free
end

"""
    instance_penalty(inst, states, ongoing_maintenance, free_days, τ)

Total penalty on day τ: sum over poles of the pole penalties.
"""
function instance_penalty(inst::Instance, states, ongoing_maintenance, free_days, τ::Int)
    return sum(pole_penalty(inst.poles[k], states[k], ongoing_maintenance[k], free_days[k], τ)
               for k in 1:inst.n_poles)
end
