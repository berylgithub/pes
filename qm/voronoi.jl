using Plots, Statistics, LaTeXStrings, LinearAlgebra

"""
dummy distance between two coordinates, should use "Mahalanobis distance" later
"""
function f_distance(x1, x2)
    return norm(x1 - x2) 
end

"""
tempoorary main container
"""
function main()
    # inputs:
    M = 10 # number of centers
    # fixed coords, ∈ (fingerprint length, data length):
    coords = Matrix{Float64}(undef, 2, 100) # a list of 2d coord arrays for testing
    # fill fixed coords:
    counter = 1
    for i ∈ 1.:10. 
        for j ∈ 1.:10.
            coords[1, counter] = i # dim1 
            coords[2, counter] = j # dim2
            counter += 1
        end
    end
    data_size = size(coords)[2] # compute once

    # Eldar's [*cite*] sampling algo:
    centers = zeros(Int64, data_size) # 1 if it is a center
    #centers[[1,3,4]] .= 1 # dum
    distances = Matrix{Float64}(undef, data_size, M) # distances from k_x, init matrix oncew
    mean_point = vec(mean(coords, dims=2)) # mean over the data for each fingerprint
    ## For all M:
    #= for m ∈ 1:M

    end =#
    ## Find largest distance:
    ### compute list of distances from mean:
    for i ∈ 1:data_size
        distances[i, 1] = f_distance(mean_point, coords[:, i])
    end
    ### sort distances descending, why sort? to avoid multiple identical centers, (NaN, inf) doesnt work:
    sorted_idx = sortperm(distances[:, 1], rev=true)
    ### check if center is already counted:
    for id ∈ sorted_idx
        if centers[id] == 0
            centers[id] = 1
            break 
        end
    end



    # transform center booleans to indexes (for other purposes, such as plot):
    idx_centers = Vector{Int64}()
    for i ∈ eachindex(centers)
        if centers[i] == 1
            push!(idx_centers, i)
        end 
    end

    # plot the points:
    scatter(coords[1, :], coords[2, :], legend = false)
    scatter!([mean_point[1,1], coords[1, centers[1]]], [mean_point[2,1], coords[2, centers[1]]], color="red")
    annotate!([mean_point[1,1]], [mean_point[2,1]].-0.25, L"$\bar w$")
    annotate!([coords[1, centers[1]]], [coords[2, centers[1]]].-0.25, L"$k_1$")
    # 
end

main()