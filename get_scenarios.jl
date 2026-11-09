using DelimitedFiles

scenarios_h = Array{Int64}(undef, 80, 6, 62)

# read txt files for each strategic period type, goup data in one file
for j in 1:6
    slice_h = readdlm("scenarios/scenarios_h$(j).txt",Int)
    scenarios_h[:, j, :] = slice_h
end

h = Array{Int64}(undef, 30*6, 80, 62)

# build scenarios, same scenarios for strategic periods with the same type
for T in 1:(30*6)

    for s in 1:80
        type = mod(T,6)
        if mod(T,6) == 0
            type = 6
        end
        h[T,s,:] = scenarios_h[s,type,:] # we use the same scenarios for each jan/fev strategic period
    end

end

