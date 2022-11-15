#include <small-matinv.hpp>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <omp.h>

#ifdef HAVE_MKL
#include <mkl.h>
#endif

#define CPU_clockrate 3.3e9

template <class Real, long q, long VecLen> void benchmark(const long N) {
  const long repeat = (long)1e9/N/(q*q*q);
  double T; // timer

  // Allocate and initialize arrays
  Real* M = (Real*)std::malloc(N*q*q*sizeof(Real));
  Real* Minv = (Real*)std::malloc(N*q*q*sizeof(Real));
  for (long i = 0; i < N*q*q; i++) M[i] = (Real)(drand48() - 0.5);

  auto print_err = [](const Real* M, const Real* Minv, const long N) {
    Real max_err = 0;
    for (long i = 0; i < N; i++) {
      for (long j0 = 0; j0 < q; j0++) {
        for (long j1 = 0; j1 < q; j1++) {
          Real MMinv = 0;
          for (long k = 0; k < q; k++) {
            MMinv += M[(i*q+j0)*q+k] * Minv[(i*q+k)*q+j1];
          }
          max_err = std::max<Real>(max_err, std::fabs(MMinv - (j0==j1?1:0)));
        }
      }
    }
    std::cout<<"        Error = "<<max_err<<'\n';
  };

  { // Naive inverse
    std::cout << "\n    Naive-inverse:\n";
    for (long i = 0; i < N*q*q; i++) Minv[i] = 0;
    T = -omp_get_wtime();
    for (long i = 0; i < repeat; i++) {
      MatInv_naive<q>(Minv, M, N);
    }
    T += omp_get_wtime();
    std::cout << "        CPU cycles = " << T*CPU_clockrate/(N*repeat) << '\n';
    std::cout << "        FLOP rate = " << MatInvFLOP_count<q>()*N*repeat/T/1e9 << " GFLOP/s\n";
    print_err(M, Minv, N);
  }

  { // Vectorized inverse
    std::cout << "\n    Vec-inverse:\n";
    for (long i = 0; i < N*q*q; i++) Minv[i] = 0;
    T = -omp_get_wtime();
    for (long i = 0; i < repeat; i++) {
      MatInv_vec<q, VecLen>(Minv, M, N);
    }
    T += omp_get_wtime();
    std::cout << "        CPU cycles = " << T*CPU_clockrate/(N*repeat) << '\n';
    std::cout << "        FLOP rate = " << MatInvFLOP_count<q>()*N*repeat/T/1e9 << " GFLOP/s\n";
    print_err(M, Minv, N);
  }

  { // Packed and vectorized inverse (when arrays are packed in vector format)
    std::cout << "\n    Packed-vec-inverse:\n";
    for (long i = 0; i < N*q*q; i++) Minv[i] = 0;
    long Npack = PackedSize(N*q*q, VecLen, q*q);
    Real* M_pack = (Real*)std::aligned_alloc(VecLen*sizeof(Real), Npack*sizeof(Real));
    Real* Minv_pack = (Real*)std::aligned_alloc(VecLen*sizeof(Real), Npack*sizeof(Real));
    VecPack<q*q, VecLen>(M_pack, Npack, M, N*q*q); // pack matrices in vector format

    T = -omp_get_wtime();
    for (long i = 0; i < repeat; i++) {
      MatInv_packed<q, VecLen>(Minv_pack, M_pack, Npack/(q*q));
    }
    T += omp_get_wtime();

    VecUnpack<q*q,VecLen>(Minv, N*q*q, Minv_pack, Npack); // unpack the inverse
    std::cout << "        CPU cycles = " << T*CPU_clockrate/(N*repeat) << '\n';
    std::cout << "        FLOP rate = " << MatInvFLOP_count<q>()*N*repeat/T/1e9 << " GFLOP/s\n";
    print_err(M, Minv, N);
    std::free(Minv_pack);
    std::free(M_pack);
  }

  #ifdef HAVE_MKL
  { // MKL
    std::cout << "\n    MKL\n";

    MKL_INT info;
    std::vector<Real*> M_ptr(N), Minv_ptr(N);
    for (long i = 0; i < N; i++) M_ptr[i] = M + i*q*q;
    for (long i = 0; i < N; i++) Minv_ptr[i] = Minv + i*q*q;

    long Npack;
    MKL_COMPACT_PACK format = mkl_get_format_compact();
    if constexpr (std::is_same<Real,double>::value) {
      Npack = mkl_dget_size_compact(q, q, format, N) / sizeof(Real);
    } else {
      Npack = mkl_sget_size_compact(q, q, format, N) / sizeof(Real);
    }
    Real* M_pack = (Real*)std::aligned_alloc(VecLen*sizeof(Real), Npack*sizeof(Real));
    Real* work = (Real*)std::aligned_alloc(VecLen*sizeof(Real), Npack*sizeof(Real));

    T = -omp_get_wtime();
    for (long j = 0; j < repeat; j++) {
      if constexpr (std::is_same<Real,double>::value) {
        mkl_dgepack_compact(MKL_ROW_MAJOR, q, q, &M_ptr[0], q, M_pack, q, format, N);
        mkl_dgetrfnp_compact(MKL_ROW_MAJOR, q, q, M_pack, q, &info, format, N);
        mkl_dgetrinp_compact(MKL_ROW_MAJOR, q, M_pack, q, work, Npack, &info, format, N);
        mkl_dgeunpack_compact(MKL_ROW_MAJOR, q, q, &Minv_ptr[0], q, M_pack, q, format, N);
      } else {
        mkl_sgepack_compact(MKL_ROW_MAJOR, q, q, &M_ptr[0], q, M_pack, q, format, N);
        mkl_sgetrfnp_compact(MKL_ROW_MAJOR, q, q, M_pack, q, &info, format, N);
        mkl_sgetrinp_compact(MKL_ROW_MAJOR, q, M_pack, q, work, Npack, &info, format, N);
        mkl_sgeunpack_compact(MKL_ROW_MAJOR, q, q, &Minv_ptr[0], q, M_pack, q, format, N);
      }
    }
    T += omp_get_wtime();

    std::cout << "        CPU cycles = " << T*CPU_clockrate/(N*repeat) << '\n';
    std::cout << "        FLOP rate = " << MatInvFLOP_count<q>()*N*repeat/T/1e9 << " GFLOP/s\n";
    print_err(M, Minv, N);
    std::free(M_pack);
    std::free(work);
  }
  #endif

  // Free memory
  std::free(M);
  std::free(Minv);
}

int main(int argc, char** argv) {
  const long N = 128; // Number of matrices in the batch

  std::cout<<"\n\n\n3x3 float Matrices:\n";
  benchmark<float, 3, DefaultVecLen<float>()>(N);

  std::cout<<"\n\n\n3x3 double Matrices:\n";
  benchmark<double, 3, DefaultVecLen<double>()>(N);

  std::cout<<"\n\n\n4x4 float Matrices:\n";
  benchmark<float, 4, DefaultVecLen<float>()>(N);

  std::cout<<"\n\n\n4x4 double Matrices:\n";
  benchmark<double, 4, DefaultVecLen<double>()>(N);

  std::cout<<"\n\n\n";
  return 0;
}

