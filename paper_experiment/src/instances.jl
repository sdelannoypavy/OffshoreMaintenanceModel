# instances.jl
#
# Instances of the numerical experiments of paper.tex (Section "Numerical experiments").
#
#   - Full model (Section "Full model"): S ∈ {1, 2, 3} substations of 9 components, with the
#     parameters of Table "tab:component-parameters", an annual quota of K0 = 30 penalty-free days
#     per substation, and the true cost of Section "Problem statement".
#   - Simplified model (Section "Simplified model"): S ∈ {1, 2, 3} substations of 7 components, with
#     the MTBF of Table "tab:mtbf-simplified", no quota (K0 = 0), and a cost that decomposes exactly
#     into local costs and a maintenance cost.
#
# In the code, a substation is called a "pole". Every substation of an instance is a copy of the
# same substation. The degradation of every component is a Markov chain, built below from its MTBF.

# ─────────────────────────────────────────────────────────────────────────────
# Time and discretisation
# ─────────────────────────────────────────────────────────────────────────────

const T_PERIOD = 60              # days in a period of two months
const N_PERIODS_PER_YEAR = 6
const N_YEARS = 20               # length of the concession (T = 120 periods)
const LEAD_TIME = 15             # lead time L: no repair can start during the first L days of a period
const FIRST_START_DAY = LEAD_TIME + 1

# The quota, the numbers of declared days and the start days are discretised in steps of STEP days
# (Section "Full model"): k, κ ∈ {0, 15, 30} and start days in {16, 31, 46}. Since every repair
# duration is a multiple of STEP, a period is split into 4 blocks of 15 days (the lead time and the
# 3 slots of the start-day grid).
const STEP = 15
"Admissible values 0, STEP, 2 STEP, … up to `top` (remaining quotas k and numbers of declared days κ)."
qrange(top::Int) = 0:STEP:top

const WORKFORCE = 1.0                         # technician capacity R (100%)
const DAILY_PENALTY = fill(1.0, T_PERIOD)     # penalty per day and per unit of capacity loss (fixes the unit of cost)

# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

"One substation: the data of its components, indexed by c."
struct Pole
    n::Vector{Int}                       # number of degradation states N_m
    d::Vector{Int}                       # repair duration d_m (days)
    P_daily::Vector{Matrix{Float64}}     # daily transition matrix, P[to, from]
    P_period::Vector{Matrix{Float64}}    # transition matrix over one period if not maintained (P_daily^60)
    P_evac::Matrix{Float64}              # P_evac[c, x]: capacity (%) of the substation when c is in state x
    technician_share::Vector{Float64}    # technician share ρ_m
    replacement_cost::Vector{Float64}    # replacement cost c^rep_m
    free_quota::Int                      # annual quota K0 of penalty-free days
    daily_penalty::Vector{Float64}       # penalty π_τ per unit of capacity loss on day τ
end

struct Instance
    name::String
    n_poles::Int                         # number of substations S
    poles::Vector{Pole}
    R::Float64                           # technician capacity, shared by all the substations
    T_period::Int
    n_years::Int
    simplified::Bool                     # true for the simplified model
end

# ─────────────────────────────────────────────────────────────────────────────
# Markov chains of degradation (Section "Setup", paragraph "Instances")
# ─────────────────────────────────────────────────────────────────────────────

"Component without subcomponents, 3 states: the expected times spent in states 1 and 2 are equal and sum to the MTBF."
function chain_three_states(MTBF::Float64)
    p = (2 / MTBF) / 365
    return [1-p 0 0; p 1-p 0; 0 p 1]
end

"Converter: 12 states, one transition per lost group of submodules, 11 transitions before failure."
function chain_converter(MTBF::Float64; n_states::Int = 12)
    p = (11 / MTBF) / 365
    P = zeros(n_states, n_states)
    for i in 1:(n_states - 1)
        P[i, i] = 1 - p; P[i + 1, i] = p
    end
    P[n_states, n_states] = 1.0
    return P
end

"Fans: 8 states (number of broken fans beyond the redundant ones), 7 transitions before failure."
function chain_fans(MTBF::Float64; n_states::Int = 8)
    p = (7 / MTBF) / 365
    P = zeros(n_states, n_states)
    for i in 1:(n_states - 1)
        P[i, i] = 1 - p; P[i + 1, i] = p
    end
    P[n_states, n_states] = 1.0
    return P
end

"Two identical units, one of which is redundant (failure probability q per unit and per day): states 2 units, 1 unit, 0 unit."
redundant_pair(q::Float64) = [1-2q*(1-q) 0 0; 2q*(1-q) 1-q 0; 0 q 1]

"Pump system (two pumps, one redundant). The daily failure probability 9.55e-5 of a pump gives an MTBF of 43 years."
chain_pumps(MTBF::Float64) = redundant_pair(9.55e-5 * 43 / MTBF)

"Transformers (two transformers, one redundant)."
chain_transformers(MTBF::Float64) = redundant_pair(1 / (365 * MTBF))

"Capacity (%) of the substation in every state of a component: 100% until the last state, 0% in the last state."
capacity_profile(n_states::Int) = [x < n_states ? 100.0 : 0.0 for x in 1:n_states]

"Fans: the capacity decreases linearly once the failures exceed the redundancy."
const CAPACITY_FANS = [100.0, 100.0, 100.0, 80.0, 60.0, 40.0, 20.0, 0.0]

"Substation built from the list of its components (named tuples with fields P, capacity, share, duration, cost)."
function build_pole(components; free_quota::Int)
    n = [size(c.P, 1) for c in components]
    P_evac = zeros(length(components), maximum(n))
    for (c, comp) in enumerate(components)
        P_evac[c, 1:n[c]] .= comp.capacity
    end
    return Pole(n, [c.duration for c in components], [c.P for c in components],
                [c.P^T_PERIOD for c in components], P_evac,
                [c.share for c in components], [c.cost for c in components],
                free_quota, copy(DAILY_PENALTY))
end

# ─────────────────────────────────────────────────────────────────────────────
# Full model: 9 components (Table "tab:component-parameters")
# ─────────────────────────────────────────────────────────────────────────────

"Substation of the full model. MTBF in years, durations in days, replacement costs in M."
function full_model_pole()
    comp(P, capacity, share, duration, cost) = (P = P, capacity = capacity, share = share, duration = duration, cost = cost)
    components = [
        comp(chain_converter(588.0),       capacity_profile(12),   1.0, 30, 2.0),    # converter
        comp(chain_three_states(43.0),     capacity_profile(3),    0.5, 15, 0.25),   # water circuit
        comp(chain_fans(43.0),             CAPACITY_FANS,          0.5, 15, 0.125),  # fans
        comp(chain_pumps(43.0),            [100.0, 90.0, 0.0],     1.0, 30, 0.25),   # pump system 1
        comp(chain_pumps(20.0),            [100.0, 90.0, 0.0],     1.0, 30, 0.25),   # pump system 2
        comp(chain_transformers(3333.0),   capacity_profile(3),    1.0, 15, 3.0),    # transformers
        comp(chain_three_states(122.0),    capacity_profile(3),    0.5, 15, 0.5),    # cooling system
        comp(chain_three_states(4762.0),   capacity_profile(3),    0.5, 15, 0.25),   # busbar
        comp(chain_three_states(20.0),     capacity_profile(3),    0.5, 15, 0.0),    # fire protection
    ]
    return build_pole(components; free_quota = 30)
end

# ─────────────────────────────────────────────────────────────────────────────
# Simplified model: 7 components (Table "tab:mtbf-simplified"), no quota
# ─────────────────────────────────────────────────────────────────────────────

"""
Substation of the simplified model: 7 of the components of the full model, with the technician shares
and the replacement costs of Table "tab:component-parameters", and the MTBF of Table "tab:mtbf-simplified".
"""
function simplified_model_pole()
    comp(P, capacity, share, duration, cost) = (P = P, capacity = capacity, share = share, duration = duration, cost = cost)
    components = [
        comp(chain_converter(10.0),        capacity_profile(12),   1.0, 30, 2.0),    # converter
        comp(chain_three_states(4.0),      capacity_profile(3),    0.5, 15, 0.25),   # water circuit
        comp(chain_fans(4.0),              CAPACITY_FANS,          0.5, 15, 0.125),  # fans
        comp(chain_pumps(4.0),             capacity_profile(3),    1.0, 15, 0.25),   # pumps
        comp(chain_transformers(12.0),     capacity_profile(3),    1.0, 15, 3.0),    # transformers
        comp(chain_three_states(6.0),      capacity_profile(3),    0.5, 15, 0.5),    # cooling system
        comp(chain_three_states(15.0),     capacity_profile(3),    0.5, 15, 0.25),   # busbar
    ]
    return build_pole(components; free_quota = 0)
end

# ─────────────────────────────────────────────────────────────────────────────
# Instances
# ─────────────────────────────────────────────────────────────────────────────

const POLE_NAMES = Dict(1 => "one_pole", 2 => "two_poles", 3 => "three_poles")

full_model_instance(S::Int) =
    Instance(POLE_NAMES[S], S, [full_model_pole() for _ in 1:S], WORKFORCE, T_PERIOD, N_YEARS, false)
simplified_model_instance(S::Int) =
    Instance(POLE_NAMES[S] * "_simple", S, [simplified_model_pole() for _ in 1:S], WORKFORCE, T_PERIOD, N_YEARS, true)

"Every instance of the paper, by name: one_pole, two_poles, three_poles (full model) and *_simple (simplified model)."
const INSTANCES = Dict(inst.name => inst for inst in
                       vcat([full_model_instance(S) for S in 1:3], [simplified_model_instance(S) for S in 1:3]))
