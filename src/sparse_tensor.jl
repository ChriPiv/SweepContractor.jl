using SparseArrays

mutable struct SparseTensor{T,N} <: AbstractArray{T,N}
    n::Int                     # number of non-zero elements
    dimensions::NTuple{N,Int32}# dimensions 
    indices::Vector{Int32}     # entry indices
    values::Vector{T}          # entry data
end


# we implement abstract array interface
# https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array

function Base.size(A::SparseTensor{T}) where T
    return A.dimensions
end

Base.firstindex(A::SparseTensor{T}) where T = 1
Base.lastindex(A::SparseTensor{T}) where T = prod(A.dimensions)

function Base.getindex(A::SparseTensor{T}, i::Int) where T
    for (j,idx) in enumerate(A.indices)
        if idx == i
            return A.values[j]
        end
    end
    return T(0)
end

function Base.setindex!(A::SparseTensor{T}, v::T, i::Int) where T
    for (j,idx) in enumerate(A.indices)
        if idx == i
            A.values[j] = v
            return
        end
    end
    push!(A.indices, i)
    push!(A.values, v)
    A.n += 1
end

function Base.getindex(A::SparseTensor{T}, I::Vararg{Int,N}) where T where N
    # TODO bound/dimension check
    idx = LinearIndices(A.dimensions)[I...]
    return Base.getindex(A, idx)
end

function Base.setindex!(A::SparseTensor{T}, v::T, I::Vararg{Int,N}) where T where N
    # TODO bound/dimension check
    idx = LinearIndices(A.dimensions)[I...]
    return Base.setindex!(A, v, idx)
end

function Base.reshape(A::SparseTensor{T}, dims::Tuple{Int64, Vararg{Int64, N}} where N) where T
    # TODO dimenison check
    B = SparseTensor{T,length(dims)}(A.n, dims, A.indices, A.values)
    return B
end


# multiplication for sparse-sparse
import Base:*
function *(A::SparseTensor{T}, B::SparseTensor{T}) where T
    # TODO bound/dimension check
    @assert length(A.dimensions) == 2
    A_csc = sparse([((idx-1)%A.dimensions[1])+1 for idx in A.indices],
                   [((idx-1)÷A.dimensions[1])+1 for idx in A.indices],
                   A.values, A.dimensions[1], A.dimensions[2])
    @assert length(B.dimensions) == 2
    B_csc = sparse([((idx-1)%B.dimensions[1])+1 for idx in B.indices],
                   [((idx-1)÷B.dimensions[1])+1 for idx in B.indices],
                   B.values, B.dimensions[1], B.dimensions[2])
    C_csc = A_csc * B_csc

    I,J,V = findnz(C_csc)
    m = length(I)
    C = SparseTensor{T,2}(m, (A.dimensions[1], B.dimensions[2]), [I[i]+A.dimensions[1]*(J[i]-1) for i=1:m], V)
    return C
end

# multiplication for sparse-dense
function *(A::SparseTensor{T}, B::Array{T}) where T
    # TODO bound/dimension check
    @assert length(A.dimensions) == 2
    A_csc = sparse([((idx-1)%A.dimensions[1])+1 for idx in A.indices],
                   [((idx-1)÷A.dimensions[1])+1 for idx in A.indices],
                   A.values, A.dimensions[1], A.dimensions[2])
    @assert length(size(B)) == 2
    C = A_csc * B
    return C
end

# multiplication for sparse-dense
function *(A::Array{T}, B::SparseTensor{T}) where T
    # TODO bound/dimension check
    @assert length(size(A)) == 2
    @assert length(B.dimensions) == 2
    B_csc = sparse([((idx-1)%B.dimensions[1])+1 for idx in B.indices],
                   [((idx-1)÷B.dimensions[1])+1 for idx in B.indices],
                   B.values, B.dimensions[1], B.dimensions[2])
    C = A * B_csc
    return C
end

function stzeros(T, size)::SparseTensor{T}
    return SparseTensor{T,length(size)}(0, size, [], [])
end

# sparse to dense
function Base.Array(A::SparseTensor{T}) where T
    dense = zeros(T, Tuple(A.dimensions))
    for i = 1:length(A.indices)
        dense[A.indices[i]] = A.values[i]
    end
    return dense
end

# permute dimensions
function Base.permutedims(A::SparseTensor{T,N}, perm) where T where N
    dest = SparseTensor{T,N}(A.n, A.dimensions, [], [])
    permutedims!(dest, A, perm)
    return dest
end

function Base.permutedims!(dest::SparseTensor{T,N}, src::AbstractArray, perm) where T where N
    dest.n = src.n
    dest.dimensions = Tuple(src.dimensions[perm[i]] for i=1:length(src.dimensions))
    dest.values = copy(src.values)

    dest.indices = zeros(Int32, dest.n)
    for i = 1:dest.n
        idx = src.indices[i]
        idx_cart = CartesianIndices(src.dimensions)[idx]
        idx_permuted = Tuple(Tuple(idx_cart)[perm[i]] for i=1:length(src.dimensions))
        dest.indices[i] = LinearIndices(dest.dimensions)[idx_permuted...]
    end
    return dest
end
