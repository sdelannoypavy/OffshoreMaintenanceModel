using JLD2

"""
    compute_optimal_maintenance_cost(c::Int, x::Int, periods_left::Int, δ, p, n)

Compute the optimal cost for component maintenance until end of year using dynamic programming.

# Arguments
- `c::Int`: Component index
- `x::Int`: Current state of the component
- `periods_left::Int`: Number of two-month periods remaining in the year (1 to 6)
- `δ`: Two-month cost array, where δ[c,x,m,κ,t] is cost for component c, state x, maintenance m
- `p`: Transition probability array, where p[c,x,k,y,k',m,κ,t] is P(x→y | maintenance m)
- `n`: Vector of state space sizes for each component

# Returns
- `cost::Float64`: Optimal cost from state x with periods_left periods remaining

# Notes
- m=1 in δ corresponds to no maintenance, m=2 corresponds to maintenance
- m=1 in p corresponds to no maintenance, m=2 corresponds to maintenance
- Uses backward dynamic programming from end of year (0 periods left) to present
- Assumes no free maintenance quota (κ=1 for both parameters)
- periods_left ranges from 1 (last period) to 6 (first period at year start)
"""
function compute_optimal_maintenance_cost(c::Int, x::Int, periods_left::Int, δ, p, n)    Q = 6    
    # Initialize DP value function
    # V[state, periods_remaining] = optimal cost from that state with periods_remaining periods left
    V = Dict{Tuple{Int, Int}, Float64}()
    
    # Base case: 0 periods left, no more cost
    for state in 1:n[c]
        V[(state, 0)] = 0.0
    end
    
    # Backward induction from Q periods down to periods_left
    for t_remain in Q:-1:periods_left
        for state in 1:n[c]
            # Try both maintenance options and choose minimum cost
            costs = Float64[]
            
            # m_idx = 1: no maintenance (m=1 in user notation)
            # m_idx = 2: maintenance (m=2 in user notation)
            for m_idx in 1:2
                # Two-month cost for this decision
                # Current time step corresponds to how many periods we've already completed
                t_abs = Q - t_remain + 1
                # κ=0 means κ+1=1 (accessing first element of κ dimension)
                two_month_cost = δ[c, state, m_idx, 1, t_abs]
                
                # Expected future cost (sum over next states)
                expected_future_cost = 0.0
                
                for next_state in 1:n[c]
                    # Transition probability from state to next_state with maintenance decision m_idx
                    # p[c, x, k, y, k', m, κ, t] where k=k'=1, κ=1
                    trans_prob = p[c, state, 1, next_state, 1, m_idx, 1, t_abs]
                    
                    # Get future cost from next_state with one less period remaining
                    future_cost = get(V, (next_state, t_remain - 1), 0.0)
                    
                    expected_future_cost += trans_prob * future_cost
                end
                
                total_cost = two_month_cost + expected_future_cost
                push!(costs, total_cost)
            end
            
            # Choose minimum cost decision
            V[(state, t_remain)] = minimum(costs)
        end
    end
    
    # Return optimal cost from initial state x with periods_left periods remaining
    return V[(x, periods_left)]
end

"""
    compute_all_optimal_costs(δ, p, n)

Precompute optimal costs for all components, states, and periods remaining.

# Returns
- `V::Dict`: Dictionary with keys (c, x, periods_left) → optimal cost
"""
function compute_all_optimal_costs(δ, p, n)
    Q = 6
    C = length(n)
    n_max = maximum(n)
    V = Dict{Tuple{Int, Int, Int}, Float64}()
    
    # Base case: 0 periods left, no cost
    for c in 1:C, state in 1:n[c]
        V[(c, state, 0)] = 0.0
    end
    
    # Backward induction: from Q periods remaining down to 1
    for t_remain in Q:-1:1
        for c in 1:C
            for state in 1:n[c]
                costs = Float64[]
                
                # Convert periods remaining to absolute time
                t_abs = Q - t_remain + 1
                
                for m_idx in 1:2
                    two_month_cost = δ[c, state, m_idx, 1, t_abs]
                    expected_future_cost = 0.0
                    
                    for next_state in 1:n[c]
                        trans_prob = p[c, state, 1, next_state, 1, m_idx, 1, t_abs]
                        future_cost = get(V, (c, next_state, t_remain - 1), 0.0)
                        expected_future_cost += trans_prob * future_cost
                    end
                    
                    push!(costs, two_month_cost + expected_future_cost)
                end
                
                V[(c, state, t_remain)] = minimum(costs)
            end
        end
    end
    
    return V
end


"""
    save_shallow_prices(V, output_file::String="shallow_prices.jld2")

Save precomputed optimal costs to JLD2 file.

# Arguments
- `V::Dict`: Dictionary of optimal costs computed by compute_all_optimal_costs
- `output_file::String`: Output filename
"""
function save_shallow_prices(V, output_file::String="shallow_prices.jld2")
    @save output_file V
    println("Shallow prices saved to $output_file")
end


"""
    one_step_cont_repair_gain(V, n)

Create a dictionary of one-step repair gains.

# Arguments
- `V::Dict`: Dictionary of optimal costs with keys (c, x, periods_left)
- `n::Vector{Int}`: Vector of state space sizes for each component

# Returns
- `gains::Dict`: Dictionary with keys (c, x, periods_left) where:
  - If periods_left > 1: value = V(c, x, periods_left) - V(c, 1, periods_left-1)
  - If periods_left == 1: value = V(c, x, 1)
  - Represents the cost benefit of performing repair (transitioning to state 1)
"""
function one_step_cont_repair_gain(V, n)    
    Q = 6    
    C = length(n)
    gains = Dict{Tuple{Int, Int, Int}, Float64}()
    
    for c in 1:C
        for x in 1:n[c]
            for m in 1:Q
                if m > 1
                    # Gain from repair: current cost minus cost of being in state 1 with one less period
                    current_cost = get(V, (c, x, m), 0.0)
                    next_state_cost = get(V, (c, 1, m - 1), 0.0)
                    gains[(c, x, m)] = current_cost - next_state_cost
                else
                    # At last period, just use the value as-is
                    gains[(c, x, m)] = get(V, (c, x, m), 0.0)
                end
            end
        end
    end
    
    return gains
end


"""
    load_and_compute_shallow_prices(delta_file::String=\"delta.jld2\", 
                                    p_file::String=\"p.jld2\",
                                    output_file::String=\"shallow_prices.jld2\",
                                    n_components::Vector{Int})

Load δ and p from files, compute optimal costs, and save to shallow_prices.jld2.

# Arguments
- `delta_file::String`: Path to delta.jld2
- `p_file::String`: Path to p.jld2
- `output_file::String`: Path to output shallow_prices.jld2
- `n_components::Vector{Int}`: Vector of state space sizes for each component

# Returns
- `V::Dict`: Dictionary with keys (c, x, periods_left) → optimal cost, where periods_left ∈ [1, 6]
"""
function load_and_compute_shallow_prices(
    delta_file::String="delta.jld2",
    p_file::String="p.jld2", 
    output_file::String="shallow_prices.jld2",
    n_components::Vector{Int}=nothing
)
    Q = 6
    
    # Load data
    @load delta_file δ
    @load p_file p
    
    # If n_components not provided, infer from δ dimensions
    if isnothing(n_components)
        C = size(δ, 1)
        n_max = size(δ, 2)
        # Assume all components have same state space size
        n_components = fill(n_max, C)
    end
    
    println("Computing optimal maintenance costs...")
    println("Components: $(length(n_components)), States: $(maximum(n_components)), Periods per year: $Q")
    
    # Compute all optimal costs
    V = compute_all_optimal_costs(δ, p, n_components)
    
    # Save results
    save_shallow_prices(V, output_file)
    
    println("Computation complete. $(length(V)) cost values computed.")
    
    return V
end


"""
    shallow_price_policy(x0, k0, t, G, n, C, d, lambda_fn)

Shallow price based maintenance policy for all components.

# Arguments
- `x0::Vector{Int}`: Current state of each component
- `k0::Int`: Current free maintenance days remaining
- `t::Int`: Current time step in year (1 to 6 periods)
- `G::Dict`: Repair gain dictionary from one_step_cont_repair_gain with keys (c, x, m)
- `n::Vector{Int}`: State space sizes for each component
- `C::Int`: Number of components
- `d::Vector{Int}`: Maintenance cost (days) for each component
- `lambda_fn::Function`: Function lambda_fn(m, k) that returns threshold for periods m and k free days

# Returns
- Tuple of (Policies_m_t_list, Policies_κ_t):
  - `Policies_m_t_list`: List of C maintenance schedules (one per component)
  - `Policies_κ_t`: Schedule for free maintenance days usage

# Algorithm
1. Compute periods remaining: m = Q - ((t-1) % 6)
2. For each component c in current state x0[c], compute G(c, x0[c], m)
3. Sum positive parts: total_gain = sum_{c} max(G(c, x0[c], m), 0)
4. If total_gain > lambda(m, k0): repair components with G > 0
5. Otherwise: no repairs
"""
function shallow_price_policy(x0, k0, t, G, n, C, d, lambda_fn, δ)
    
    # Compute periods remaining in year (assuming year starts at t and has 6 periods)
    # t ranges from 1 to 6, so periods_remaining goes from 6 (at t=1) down to 1 (at t=6)
    periods_remaining = 7 - ((t - 1) % 6) - 1
    if periods_remaining < 1
        periods_remaining = 6
    end
    
    # Get threshold for this state
    lambda = lambda_fn(periods_remaining, k0)
    πm = δ[1, 12, 2, 1, (t-1)%6 + 1]  
    η = min(πm, lambda)
    
    # Compute total repair gain
    total_gain = 0.0
    gains_by_component = Float64[]
    
    for c in 1:C
        gain_c = get(G, (c, x0[c], periods_remaining), 0.0)
        push!(gains_by_component, gain_c)
        if gain_c > 0
            total_gain += gain_c
        end
    end
    
    # Decide on repairs
    m_opt = [0 for c in 1:C]
    κ_opt = 0
    
    if total_gain > η
        # Repair components with positive gain
        for c in 1:C
            if gains_by_component[c] > 0
                m_opt[c] = 1
                if lambda < πm
                    κ_opt = k0  # Use free maintenance if threshold is below πm
                end
            end
        end
    end
    
    # Create policy schedules using the same format as benchmark policies
    # These functions are assumed to be available from elsewhere (e.g., parameters file)
    t_start = 1
    T_per = 62  # standard 2-month period
    
    Policies_m_t_list = [create_schedule_m(m_opt[c], t_start, T_per) for c in 1:C]
    Policies_κ_t = create_schedule_κ(κ_opt, t_start, T_per)
    
    return Policies_m_t_list, Policies_κ_t
end


"""
    shallow_price_policy_fixed_lambda(x0, k0, t, G, n, C, d, lambda_const=0.0)

Shallow price policy with fixed lambda threshold (no dependence on m or k).

# Arguments
- `x0::Vector{Int}`: Current state of each component
- `k0::Int`: Current free maintenance days remaining
- `t::Int`: Current time step
- `G::Dict`: Repair gain dictionary
- `n::Vector{Int}`: State space sizes
- `C::Int`: Number of components
- `d::Vector{Int}`: Maintenance cost (days) for each component
- `lambda_const::Float64`: Fixed threshold (default 0.0)

# Returns
- Tuple of (Policies_m_t_list, Policies_κ_t)
"""
function shallow_price_policy_fixed_lambda(x0, k0, t, G, n, C, d, δ, lambda_const=0.0)
    lambda_fn = (m, k) -> lambda_const
    return shallow_price_policy(x0, k0, t, G, n, C, d, lambda_fn, δ)
end
