using Random, Distributions, ProgressMeter, SparseArrays, JLD2, DataFrames

# ---------- Mobility matrix (unchanged) ----------
function make_mobility_matrix(num_zones::Int, density::Float64)
    return rand(Bernoulli(density), num_zones, num_zones)
end

# ---------- Initialisation (unchanged, returns a named tuple) ----------
function initial_state(N::Int, zone_size::Int, initial_infected::Int,
                       start_mode::Symbol; seed=nothing)
    !isnothing(seed) && Random.seed!(seed)
    num_zones_per_dim = N ÷ zone_size
    num_zones_total   = num_zones_per_dim^2
    total_nodes       = N * N
    home_x   = Vector{Int}(undef, total_nodes)
    home_y   = Vector{Int}(undef, total_nodes)
    zone_id  = Vector{Int}(undef, total_nodes)
    idx = 0
    for x in 1:N, y in 1:N
        idx += 1
        home_x[idx]  = x
        home_y[idx]  = y
        zx = (x-1) ÷ zone_size + 1
        zy = (y-1) ÷ zone_size + 1
        zone_id[idx] = (zx-1)*num_zones_per_dim + zy
    end
    infected = falses(total_nodes)
    if start_mode == :single_zone
        z = rand(1:num_zones_total)
        nodes_in_zone = findall(==(z), zone_id)
        n_infect = min(initial_infected, length(nodes_in_zone))
        infected[sample(nodes_in_zone, n_infect; replace=false)] .= true
    elseif start_mode == :random
        infected[sample(1:total_nodes, initial_infected; replace=false)] .= true
    else
        error("start_mode must be :single_zone or :random")
    end
    return (; home_x, home_y, zone_id, infected, N, zone_size,
              num_zones_per_dim, num_zones_total)
end

# ---------- Infection phase – now returns per‑zone new infections ----------
function apply_infection!(infected, cur_x, cur_y, beta, offsets, zone_id, periodic, N)
    total = length(infected)
    grid = [Vector{Int}() for _ in 1:N, _ in 1:N]
    for i in 1:total
        push!(grid[cur_x[i], cur_y[i]], i)
    end
    new_infected = falses(total)
    for i in findall(infected)
        cx, cy = cur_x[i], cur_y[i]
        for (dx, dy) in offsets
            nx = periodic ? mod1(cx + dx, N) : cx + dx
            ny = periodic ? mod1(cy + dy, N) : cy + dy
            if !periodic && (nx < 1 || nx > N || ny < 1 || ny > N)
                continue
            end
            for j in grid[nx, ny]
                if !infected[j] && !new_infected[j] && rand() < beta
                    new_infected[j] = true
                end
            end
        end
    end
    infected .|= new_infected
    # Count new infections per home zone
    zone_new = zeros(Int, maximum(zone_id))
    for i in findall(new_infected)
        zone_new[zone_id[i]] += 1
    end
    return zone_new
end

# ------------------------------------------------------------
# Generate a deterministic OD flow matrix based on zone
# populations and the binary mobility matrix A.
# ------------------------------------------------------------
function make_deterministic_OD!(OD_flow, zones_nodes, A, p_move, num_zones)
    for i in 1:num_zones
        pop_i = length(zones_nodes[i])
        n_move = round(Int, p_move * pop_i)
        if n_move == 0
            continue
        end
        allowed = findall(>(0), A[i, :])
        isempty(allowed) && continue
        # Distribute n_move uniformly among allowed destinations
        n_dest = length(allowed)
        base = n_move ÷ n_dest
        remainder = n_move % n_dest
        for (k, j) in enumerate(allowed)
            OD_flow[i, j] = base + (k <= remainder ? 1 : 0)
        end
    end
end

# ------------------------------------------------------------
# Revised step! – uses a fixed OD_flow matrix
# ------------------------------------------------------------
function step!(state, A, OD_flow, p_move, beta_day, beta_night, gamma; periodic)
    total_nodes = state.N^2
    num_zones = state.num_zones_total
    cur_x = copy(state.home_x)
    cur_y = copy(state.home_y)

    # ---- Movement using the fixed OD_flow ----
    zones_nodes = [Int[] for _ in 1:num_zones]
    for i in 1:total_nodes
        push!(zones_nodes[state.zone_id[i]], i)
    end

    for i in 1:num_zones
        pop_i = length(zones_nodes[i])
        n_move = round(Int, p_move * pop_i)
        n_move == 0 && continue
        
        # Randomly select which individuals move (same number every day)
        movers = sample(zones_nodes[i], n_move; replace=false)
        # Partition them into destination bins according to OD_flow[i, :]
        dest_counts = OD_flow[i, :]   # a SparseVector or Vector
        # We need to assign exactly OD_flow[i,j] movers to each j
        # First, create a shuffled list of movers
        shuffled = shuffle(movers)
        idx = 1
        for j in findall(>(0), dest_counts)
            n_to_j = dest_counts[j]
            n_to_j == 0 && continue
            # Take n_to_j movers from the shuffled list
            for _ in 1:n_to_j
                node = shuffled[idx]
                idx += 1
                # Place node randomly inside destination zone j
                zx = (j-1) ÷ state.num_zones_per_dim + 1
                zy = (j-1) % state.num_zones_per_dim + 1
                cur_x[node] = (zx-1)*state.zone_size + rand(1:state.zone_size)
                cur_y[node] = (zy-1)*state.zone_size + rand(1:state.zone_size)
            end
        end
    end

    # ---- Day infection (unchanged) ----
    offsets_day = [(dx, dy) for dx in -2:2 for dy in -2:2 if !(dx==0 && dy==0)]
    new_day = apply_infection!(state.infected, cur_x, cur_y, beta_day, offsets_day,
                               state.zone_id, periodic, state.N)

    # ---- Night: return home (unchanged) ----
    cur_x .= state.home_x
    cur_y .= state.home_y
    offsets_night = [(1,0), (-1,0), (0,1), (0,-1)]
    new_night = apply_infection!(state.infected, cur_x, cur_y, beta_night, offsets_night,
                                 state.zone_id, periodic, state.N)

    zone_new = new_day .+ new_night

    # ---- Recovery (unchanged) ----
    for i in 1:total_nodes
        if state.infected[i] && rand() < gamma
            state.infected[i] = false
        end
    end
    return zone_new   # only return new infections per zone (no OD matrix needed)
end

# ------------------------------------------------------------
# Revised simulate_and_save – OD_flow created once
# ------------------------------------------------------------
function simulate_and_save(state, A, p_move, beta_day, beta_night, gamma;
                           days, warmup=0, periodic=true,
                           out_prefix = "sim")
    num_zones = state.num_zones_total
    total_nodes = state.N^2
    
    # Precompute zone member lists (static, can be reused)
    zones_nodes = [Int[] for _ in 1:num_zones]
    for i in 1:total_nodes
        push!(zones_nodes[state.zone_id[i]], i)
    end
    
    # ---- Build the deterministic OD flow matrix ----
    OD_flow = spzeros(Int, num_zones, num_zones)
    make_deterministic_OD!(OD_flow, zones_nodes, A, p_move, num_zones)
    # (We will save this matrix once, not every day)
    
    # Data storage
    new_inf = zeros(Int, num_zones, days + 1)
    zone_ever_infected = falses(num_zones)
    
    total_steps = warmup + days
    p = Progress(total_steps, dt=1.0, desc="Simulating: ")
    for d in 1:total_steps
        zone_new = step!(state, A, OD_flow, p_move, beta_day, beta_night, gamma; periodic)
        if d > warmup
            day_idx = d - warmup
            new_inf[:, day_idx + 1] = zone_new
            zone_ever_infected[zone_new .> 0] .= true
        end
        next!(p)
    end
    
    num_infected_zones = count(zone_ever_infected)
    jldsave(out_prefix * "_new_infections.jld2";
            new_inf = new_inf,
            num_communities = num_zones,
            zone_size = state.zone_size,
            N = state.N,
            beta_day = beta_day,
            beta_night = beta_night,
            p_move = p_move,
            gamma = gamma,
            mobility_matrix = A,
            OD_flow = OD_flow,          # the single fixed OD matrix
            num_infected_zones = num_infected_zones,
            zone_ever_infected = zone_ever_infected)
    println("Data saved with prefix: $out_prefix")
    return nothing
end


const N                   = 1024          # grid size
const ZONE_SIZE           = 16           # zone side length
const INITIAL_INFECTED    = 10          # initially infected
const START_MODE          = :single_zone     # or :single_zone
const P_MOVE              = 0.5         # fraction moving per day
const GAMMA               = 0.3         # recovery probability
const DAYS                = 200         # recorded days (after warm‑up)
const WARMUP              = 50          # burn‑in
const PERIODIC            = false     # torus boundaries
const SEED                = 1234        # set to nothing if you don't want reproducibility

# ------------------------------------------------------------
# 3) Sweep ranges
# ------------------------------------------------------------
density_vals = 0.0:0.1:1.0
beta_vals    = 0.0:0.1:1.0

# ------------------------------------------------------------
# 4) Prepare result table
# ------------------------------------------------------------
results = DataFrame(
    density        = Float64[],
    beta           = Float64[],
    mean_incidence = Float64[],
    infected_zones = Int64[]
)

# ------------------------------------------------------------
# 5) Run the sweep
# ------------------------------------------------------------
@showprogress "Running parameter sweep..." for density in density_vals
    num_zones = (N ÷ ZONE_SIZE)^2
    A = make_mobility_matrix(num_zones, density)

    for beta in beta_vals
        # Unique prefix for each combination – all files go into a subfolder
        prefix = "output/d$(density)_b$(beta)"
        mkpath(prefix)   # create folder if not present

        state = initial_state(N, ZONE_SIZE, INITIAL_INFECTED, START_MODE; seed=SEED)

        # Run the simulation (saves OD and new_infections files)
        simulate_and_save(state, A, P_MOVE, beta, beta, GAMMA;
                          days = DAYS, warmup = WARMUP, periodic = PERIODIC,
                          out_prefix = prefix)

        # Load the new infections data that was just saved
        jld_path = prefix * "_new_infections.jld2"
        data = load(jld_path)
        new_inf = data["new_inf"]   # size (num_zones, days+1)

        # Compute average daily incidence (fraction of total population)
        total_pop = N * N
        daily_tot = vec(sum(new_inf, dims=1))   # total new infections each day
        # column 1 is day 0 (all zeros), so use columns 2:end
        mean_inc = sum(daily_tot[2:end]) / (DAYS * total_pop)

        # Number of zones that ever had an infection
        n_inf_zones = data["num_infected_zones"]

        push!(results, (density, beta, mean_inc, n_inf_zones))
    end
end

# ------------------------------------------------------------
# 6) Save summary CSV
# ------------------------------------------------------------
CSV.write("sweep_summary.csv", results)
println("Summary saved to sweep_summary.csv")


