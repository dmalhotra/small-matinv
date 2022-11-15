#ifndef _SMALL_MATINV_HPP_
#define _SMALL_MATINV_HPP_

#include <vectorclass.h>

#if defined(__AVX512__) || defined(__AVX512F__)
template <class ScalarType> constexpr int DefaultVecLen() { return 64/sizeof(ScalarType); }
#elif defined(__AVX__)
template <class ScalarType> constexpr int DefaultVecLen() { return 32/sizeof(ScalarType); }
#elif defined(__SSE4_2__)
template <class ScalarType> constexpr int DefaultVecLen() { return 16/sizeof(ScalarType); }
#else
template <class ScalarType> constexpr int DefaultVecLen() { return 1; }
#endif

/**
 * Returns the number of floating-point operations in computing a matrix
 * inverse.
 */
template <long q> constexpr long MatInvFLOP_count();


/**
 * Compute batched small matrix inverse.
 *
 * \param[out] Minv the output array with matrix inverses.
 *
 * \prarm[in] M the input array with matrices.
 *
 * \param[in] N number of matrices.
 *
 * \tparam q the dimension of the matrices.
 *
 * \tparam Real the type of scalar data.
 */
template <long q, class Real> static void MatInv_naive(Real* Minv, const Real* M, long N);


/**
 * Compute batched small matrix inverse using vector intrinsics.
 *
 * \param[out] Minv the output array with matrix inverses.
 *
 * \prarm[in] M the input array with matrices.
 *
 * \param[in] N number of matrices.
 *
 * \tparam q the dimension of the matrices.
 *
 * \tparam VecLen the number of scalar elements in a SIMD vector. Recommended
 * vector length is given by DefaultVecLen<Real>().
 *
 * \tparam Real the type of scalar data.
 */
template <long q, long VecLen, class Real> static void MatInv_vec(Real* Minv, const Real* M, long N);


/**
 * Returns the number of elements in packed vectorizable format. Similar to
 * mkl_dget_size_compact(q, q, format, N).
 *
 * \param[in] N the number of elements in the array.
 *
 * \tparam VecLen the number of scalar elements in a SIMD vector. Recommended
 * vector length is given by DefaultVecLen<Real>().
 *
 * \tparam qq the dimension of the data (q*q for q-by-q matrices).
 */
constexpr long PackedSize(long N, long VecLen, long qq);

/**
 * Pack data to vectorizable format. Similar to mkl_dgepack_compact().
 *
 * \param[out] Y pointer to the packed output array.
 *
 * \param[in] Ny the number of elements in array Y. It should match the result
 * of PackedSize(Nx, VecLen, qq).
 *
 * \param[in] X pointer to the input array.
 *
 * \param[in] Nx the number of elements in array X.
 *
 * \tparam qq the dimension of the data (q*q for q-by-q matrices).
 *
 * \tparam VecLen the number of scalar elements in a SIMD vector. Recommended
 * vector length is given by DefaultVecLen<Real>().
 *
 * \tparam Real the type of scalar data.
 */
template <long qq, long VecLen, class Real> static void VecPack(Real* Y, long Ny, const Real* X, long Nx);

/**
 * Compute small matrix inverse when arrays are in packed vector format.
 *
 * \param[out] Minv the output array with matrix inverses in packed format.
 *
 * \prarm[in] M the input array with matrices in packed format.
 *
 * \param[in] N number of matrices.
 *
 * \tparam q the dimension of the matrices.
 *
 * \tparam VecLen the number of scalar elements in a SIMD vector. Recommended
 * vector length is given by DefaultVecLen<Real>().
 *
 * \tparam Real the type of scalar data.
 */
template <long q, long VecLen, class Real> static void MatInv_packed(Real* Minv, const Real* M, long N);

/**
 * Unpack data from vectorizable format. Similar to mkl_dgeunpack_compact().
 *
 * \param[out] X pointer to the output array.
 *
 * \param[in] Nx the number of elements in array X.
 *
 * \param[in] Y pointer to the packed input array.
 *
 * \param[in] Ny the number of elements in array Y. It should match the result
 * of PackedSize(Nx, VecLen, qq).
 *
 * \tparam qq the dimension of the data (q*q for q-by-q matrices).
 *
 * \tparam VecLen the number of scalar elements in a SIMD vector. Recommended
 * vector length is given by DefaultVecLen<Real>().
 *
 * \tparam Real the type of scalar data.
 */
template <long qq, long VecLen, class Real> static void VecUnpack(Real* X, long Nx, const Real* Y, long Ny);


#include <small-matinv.cpp>

#endif //_SMALL_MATINV_HPP_
