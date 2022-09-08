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
    M = 2 # number of centers
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
    mean_distances = Vector{Float64}(undef, data_size) # distances from mean point
    distances = Matrix{Float64}(undef, data_size, M) # distances from k_x, init matrix oncew
    ## Start from mean of all points:
    mean_point = vec(mean(coords, dims=2)) # mean over the data for each fingerprint
    ref_point = mean_point # init with mean, then k_x next iter
    for i ∈ 1:data_size
        mean_distances[i] = f_distance(ref_point, coords[:, i])
    end
    ### get point with max distance from mean:
    _, selected_id = findmax(mean_distances)
    centers[selected_id] = 1

    ## To find k_x s.t. x > 1, for m ∈ M:
#=     for m ∈ 1:M
        ## Find largest distance:
        ### compute list of distances from mean:
        for i ∈ 1:data_size
            distances[i, m] = f_distance(ref_point, coords[:, i])
        end
        ### take the column minimum for each row:
        min_dist = Vector{Float64}(undef, data_size)
        for i ∈ 1:data_size
            min_dist[i] = minimum(distances[i, 1:m])
        end
        println(min_dist)
        ### sort distances descending, why sort? to avoid multiple identical centers, (NaN, inf) doesnt work:
        sorted_idx = sortperm(min_dist, rev=true)
        ### check if center is already counted:
        selected_id = 0
        for id ∈ sorted_idx
            if centers[id] == 0
                centers[id] = 1
                selected_id = id
                break 
            end
        end
        ### reassign ref point by the new center:
        ref_point = coords[:, selected_id]

    end
    

    # transform center booleans to indexes (for other purposes, such as plot):
    idx_centers = Vector{Int64}()
    for i ∈ eachindex(centers)
        if centers[i] == 1
            push!(idx_centers, i)
        end 
    end
 =#

    # plot the points:
    scatter(coords[1, :], coords[2, :], legend = false)
    scatter!([mean_point[1,1], coords[1, centers[1]]], [mean_point[2,1], coords[2, centers[1]]], color="red")
    annotate!([mean_point[1,1]], [mean_point[2,1]].-0.25, L"$\bar w$")
    annotate!([coords[1, centers[1]]], [coords[2, centers[1]]].-0.25, L"$k_1$")
    # 
end

main()