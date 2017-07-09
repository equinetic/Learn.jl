# Converts a vector of labels into a matrix of boolean encodings
function onehot(x::AbstractVector)::AbstractVecOrMat
  labels = sort!(label(x))

  if nlabel(labels)==2
    return x
  end

  mat = Matrix{Int}(size(x,1), nlabel(labels))
  mat .= x .== labels'
  mat
end

# Converts a matrix of boolean encodings to a vector of labels
function unhot(x::AbstractVecOrMat, labels::AbstractVector)::AbstractVector
  if size(x,2)==1
    return x
  end

  im = [indmax(x[i,:]) for i=1:size(x,1)]
  labels[im]
end
