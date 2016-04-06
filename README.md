# TensorDecompositions

[![Build Status](https://travis-ci.org/mhauru/TensorDecompositions.jl.svg?branch=master)](https://travis-ci.org/mhauru/TensorDecompositions.jl)

TensorDecompositoins is a Julia library for eigenvalue and singular value
decompositions of tensors.

## Installing
`Pkg.clone("git://github.com/mhauru/TensorDecompositions.jl.git")`

## Exported functions

#### `tensorsvd`

```julia
tensorsvd(A, a, b;
          chis=nothing, eps=0,
          return_error=false, print_error=false,
          break_degenerate=false, degeneracy_eps=1e-6,
          norm_type=:frobenius)
```
Singular valued decomposes a tensor `A`. The indices of `A` are
permuted so that the indices listed in the Array/Tuple `a` are on the "left"
side and indices listed in `b` are on the "right".  The resulting tensor is
then `reshape`d to a matrix, and this matrix is SVDed into `U*diagm(S)*Vt`.
Finally, the unitary matrices `U` and `Vt` are `reshape`d to tensors so that
they have a new index coming from the SVD, for `U` as the last index and for
`Vt` as the first, and `U` has indices `a` as its first indices and `V` has
indices `b` as its last indices.

If `eps>0` then the SVD may be truncated if the relative error can be kept
below `eps`. For this purpose different dimensions to truncate to can be tried,
and these dimensions should be listed in `chis`. If `chis` is `nothing` (the
default) then the full range of possible dimensions is tried. If
`break_degenerate=false` (the default) then the truncation never cuts between
degenerate singular values. `degeneracy_eps` controls how close the values need
to be to be considered degenerate.

`norm_type` specifies the norm used to measure the error. This defaults to
`:frobenius`, which means that the error measured is the Frobenius norm of the
difference between `A` and the decomposition, divided by the Frobenius norm of
`A`.  This is the same thing as the 2-norm of the singular values that are
truncated out, divided by the 2-norm of all the singular values. The other
option is `:trace`, in which case a 1-norm is used instead.

If `print_error=true` the truncation error is printed. The default is `false`.

If `return_error=true` then the truncation error is also returned.
The default is `false`.

Note that no iterative techniques are used, which means choosing to truncate
provides no performance benefits: The full SVD is computed in any case.

Output is `U`, `S`, `Vt`, and possibly `error`. Here `S` is a vector of
singular values and `U` and `Vt` are isometric tensors (unitary if the matrix
that is SVDed is square and there is no truncation) such that  `U*diag(S)*Vt =
A`, up to truncation errors.

#### `tensoreig`

```julia
tensoreig(A, a, b;
          chis=nothing, eps=0,
          return_error=false, print_error=false,
          break_degenerate=false, degeneracy_eps=1e-6,
          norm_type=:frobenius,
          hermitian=false)
```
Finds the "right" eigenvectors and eigenvalues of `A`. The indices of `A` are
permuted so that the indices listed in the Array/Tuple `a` are on the "left"
side and indices listed in `b` are on the "right".  The resulting tensor is
then `reshape`d to a matrix, and `eig` is called on this matrix to get the
vector of eigenvalues `E` and matrix of eigenvectors `U`. Finally, `U` is
reshaped to a tensor that has as its last index the one that enumerates the
eigenvectors and the indices in `a` as its first indices.

Truncation and error printing work as with `tensorsvd`.

Note that no iterative techniques are used, which means that choosing to
truncate provides no performance benefits: All the eigenvalues are computed
in any case.

The keyword argument `hermitian` (`false` by default) tells the algorithm
whether the reshaped matrix is Hermitian or not. If `hermitian=true`, then `A =
U*diagm(E)*U'` up to the truncation error.

Output is `E`, `U`, and possibly `error`, if `return_error=true`. Here `E` is a
vector of eigenvalues values and `U[:,...,:,k]` is the kth eigenvector.

#### `tensorsplit`

```julia
tensorsplit(A, a, b; kwargs...)
```
Calls `tensorsvd` with the arguments given to it to decompose the given tensor
`A` with indices `a` on one side and indices `b` on the other.  It then splits
the diagonal matrix of singular values into two with a square root and
multiplies these weights into the isometric tensors.  Thus `tensorsplit` ends
up splitting `A` into two parts, which are then returned, possibly together
with auxiliary data such as a truncation error. If the keyword argument
`hermitian=true`, an eigenvalue decomposition is used in stead of an SVD. All
the keyword arguments are passed to either `tensorsvd` or `tensoreig`.

See `tensorsvd` and `tensoreig` for further documentation.

