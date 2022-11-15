#include <type_traits>
#include <algorithm>
#include <cassert>
#include <cmath>

template <class Real, int VecLen> class Vec {};
template <class Real> class Vec<Real,1> : Real {
  public:
  Vec() = default;
  Vec(const Real& v) : Real(v) {}
};
template <> class Vec<double,2> : public Vec2d {
  public:
  Vec() = default;
  Vec(const Vec2d& v) : Vec2d(v) {}
  Vec(double a, double b) : Vec2d(a,b) {}
};
template <> class Vec<double,4> : public Vec4d {
  public:
  Vec() = default;
  Vec(const Vec4d& v) : Vec4d(v) {}
  Vec(double a, double b, double c, double d) : Vec4d(a,b,c,d) {}
};
template <> class Vec<double,8> : public Vec8d {
  public:
  Vec() = default;
  Vec(const Vec8d& v) : Vec8d(v) {}
  Vec(double a, double b, double c, double d, double e, double f, double g, double h) : Vec8d(a,b,c,d,e,f,g,h) {}
};
template <> class Vec<float,4> : public Vec4f {
  public:
  Vec() = default;
  Vec(const Vec4f& v) : Vec4f(v) {}
  Vec(float a, float b, float c, float d) : Vec4f(a,b,c,d) {}
};
template <> class Vec<float,8> : public Vec8f {
  public:
  Vec() = default;
  Vec(const Vec8f& v) : Vec8f(v) {}
  Vec(float a, float b, float c, float d, float e, float f, float g, float h) : Vec8f(a,b,c,d,e,f,g,h) {}
};
template <> class Vec<float,16> : public Vec16f {
  public:
  Vec() = default;
  Vec(const Vec16f& v) : Vec16f(v) {}
  template <class ...T1> Vec(float a, T1... args) : Vec16f(a,args...) {}
};

template <long q, class MatIn, class MatOut> inline void MatInvKernel(MatOut& Minv, const MatIn& M) noexcept {
  using ElemType = typename std::remove_reference<decltype(Minv[0])>::type;

  ElemType Mcof[q*q];
  if constexpr (q == 3) {
    Mcof[0] = M[4]*M[8] - M[5]*M[7];
    Mcof[1] = M[2]*M[7] - M[1]*M[8];
    Mcof[2] = M[1]*M[5] - M[2]*M[4];

    Mcof[3] = M[5]*M[6] - M[3]*M[8];
    Mcof[4] = M[0]*M[8] - M[2]*M[6];
    Mcof[5] = M[2]*M[3] - M[0]*M[5];

    Mcof[6] = M[3]*M[7] - M[4]*M[6];
    Mcof[7] = M[6]*M[1] - M[0]*M[7];
    Mcof[8] = M[0]*M[4] - M[1]*M[3];

    const ElemType det_inv = ((ElemType)1) / (Mcof[0]*M[0] + Mcof[1]*M[3] + Mcof[2]*M[6]);
    for (long i = 0; i < 9; i++) Minv[i] = Mcof[i] * det_inv;
  } else if constexpr (q == 4) {
    if (0) {
      ElemType A2323 = M[10] * M[15] - M[11] * M[14] ;
      ElemType A1323 = M[9] * M[15] - M[11] * M[13] ;
      ElemType A1223 = M[9] * M[14] - M[10] * M[13] ;
      ElemType A0323 = M[8] * M[15] - M[11] * M[12] ;
      ElemType A0223 = M[8] * M[14] - M[10] * M[12] ;
      ElemType A0123 = M[8] * M[13] - M[9] * M[12] ;
      ElemType A2313 = M[6] * M[15] - M[7] * M[14] ;
      ElemType A1313 = M[5] * M[15] - M[7] * M[13] ;
      ElemType A1213 = M[5] * M[14] - M[6] * M[13] ;
      ElemType A2312 = M[6] * M[11] - M[7] * M[10] ;
      ElemType A1312 = M[5] * M[11] - M[7] * M[9] ;
      ElemType A1212 = M[5] * M[10] - M[6] * M[9] ;
      ElemType A0313 = M[4] * M[15] - M[7] * M[12] ;
      ElemType A0213 = M[4] * M[14] - M[6] * M[12] ;
      ElemType A0312 = M[4] * M[11] - M[7] * M[8] ;
      ElemType A0212 = M[4] * M[10] - M[6] * M[8] ;
      ElemType A0113 = M[4] * M[13] - M[5] * M[12] ;
      ElemType A0112 = M[4] * M[9] - M[5] * M[8] ;

      ElemType inv[16];
      inv[0] =  ( M[5] * A2323 - M[6] * A1323 + M[7] * A1223 );
      inv[1] = -( M[1] * A2323 - M[2] * A1323 + M[3] * A1223 );
      inv[2] =  ( M[1] * A2313 - M[2] * A1313 + M[3] * A1213 );
      inv[3] = -( M[1] * A2312 - M[2] * A1312 + M[3] * A1212 );

      const ElemType det_inv = ((ElemType)1) / ( M[0] * inv[0] + M[4] * inv[1] + M[8] * inv[2] + M[12] * inv[3]);

      inv[4] = -( M[4] * A2323 - M[6] * A0323 + M[7] * A0223 );
      inv[5] =  ( M[0] * A2323 - M[2] * A0323 + M[3] * A0223 );
      inv[6] = -( M[0] * A2313 - M[2] * A0313 + M[3] * A0213 );
      inv[7] =  ( M[0] * A2312 - M[2] * A0312 + M[3] * A0212 );
      inv[8] =  ( M[4] * A1323 - M[5] * A0323 + M[7] * A0123 );
      inv[9] = -( M[0] * A1323 - M[1] * A0323 + M[3] * A0123 );
      inv[10] =  ( M[0] * A1313 - M[1] * A0313 + M[3] * A0113 );
      inv[11] = -( M[0] * A1312 - M[1] * A0312 + M[3] * A0112 );
      inv[12] = -( M[4] * A1223 - M[5] * A0223 + M[6] * A0123 );
      inv[13] =  ( M[0] * A1223 - M[1] * A0223 + M[2] * A0123 );
      inv[14] = -( M[0] * A1213 - M[1] * A0213 + M[2] * A0113 );
      inv[15] =  ( M[0] * A1212 - M[1] * A0212 + M[2] * A0112 );
      for (long i = 0; i < 16; i++) Minv[i] = inv[i] * det_inv;
    } else {
      Mcof[ 0] =  M[5] * (M[10] * M[15] - M[11] * M[14]) - M[9] * (M[6] * M[15] - M[7] * M[14]) + M[13] * (M[6] * M[11] - M[7] * M[10]);
      Mcof[ 1] = -M[1] * (M[10] * M[15] - M[11] * M[14]) + M[9] * (M[2] * M[15] - M[3] * M[14]) - M[13] * (M[2] * M[11] - M[3] * M[10]);
      Mcof[ 2] =  M[1] * (M[ 6] * M[15] - M[ 7] * M[14]) - M[5] * (M[2] * M[15] - M[3] * M[14]) + M[13] * (M[2] * M[ 7] - M[3] * M[ 6]);
      Mcof[ 3] = -M[1] * (M[ 6] * M[11] - M[ 7] * M[10]) + M[5] * (M[2] * M[11] - M[3] * M[10]) - M[ 9] * (M[2] * M[ 7] - M[3] * M[ 6]);

      const ElemType det_inv = ((ElemType)1) / ( M[0] * Mcof[0] + M[4] * Mcof[1] + M[8] * Mcof[2] + M[12] * Mcof[3]);

      Mcof[ 4] = -M[4] * (M[10] * M[15] - M[11] * M[14]) + M[8] * (M[6] * M[15] - M[7] * M[14]) - M[12] * (M[6] * M[11] - M[7] * M[10]);
      Mcof[ 5] =  M[0] * (M[10] * M[15] - M[11] * M[14]) - M[8] * (M[2] * M[15] - M[3] * M[14]) + M[12] * (M[2] * M[11] - M[3] * M[10]);
      Mcof[ 6] = -M[0] * (M[ 6] * M[15] - M[ 7] * M[14]) + M[4] * (M[2] * M[15] - M[3] * M[14]) - M[12] * (M[2] * M[ 7] - M[3] * M[ 6]);
      Mcof[ 7] =  M[0] * (M[ 6] * M[11] - M[ 7] * M[10]) - M[4] * (M[2] * M[11] - M[3] * M[10]) + M[ 8] * (M[2] * M[ 7] - M[3] * M[ 6]);
      Mcof[ 8] =  M[4] * (M[ 9] * M[15] - M[11] * M[13]) - M[8] * (M[5] * M[15] - M[7] * M[13]) + M[12] * (M[5] * M[11] - M[7] * M[ 9]);
      Mcof[ 9] = -M[0] * (M[ 9] * M[15] - M[11] * M[13]) + M[8] * (M[1] * M[15] - M[3] * M[13]) - M[12] * (M[1] * M[11] - M[3] * M[ 9]);
      Mcof[10] =  M[0] * (M[ 5] * M[15] - M[ 7] * M[13]) - M[4] * (M[1] * M[15] - M[3] * M[13]) + M[12] * (M[1] * M[ 7] - M[3] * M[ 5]);
      Mcof[11] = -M[0] * (M[ 5] * M[11] - M[ 7] * M[ 9]) + M[4] * (M[1] * M[11] - M[3] * M[ 9]) - M[ 8] * (M[1] * M[ 7] - M[3] * M[ 5]);
      Mcof[12] = -M[4] * (M[ 9] * M[14] - M[10] * M[13]) + M[8] * (M[5] * M[14] - M[6] * M[13]) - M[12] * (M[5] * M[10] - M[6] * M[ 9]);
      Mcof[13] =  M[0] * (M[ 9] * M[14] - M[10] * M[13]) - M[8] * (M[1] * M[14] - M[2] * M[13]) + M[12] * (M[1] * M[10] - M[2] * M[ 9]);
      Mcof[14] = -M[0] * (M[ 5] * M[14] - M[ 6] * M[13]) + M[4] * (M[1] * M[14] - M[2] * M[13]) - M[12] * (M[1] * M[ 6] - M[2] * M[ 5]);
      Mcof[15] =  M[0] * (M[ 5] * M[10] - M[ 6] * M[ 9]) - M[4] * (M[1] * M[10] - M[2] * M[ 9]) + M[ 8] * (M[1] * M[ 6] - M[2] * M[ 5]);
      for (long i = 0; i < 16; i++) Minv[i] = Mcof[i] * det_inv;
    }
  } else {
    static_assert(q!=q, "Not implemented");
  }
}

template <long VecLen, class VecArray> inline void VecTranspose(VecArray& u) noexcept {
  using Vec = typename std::remove_reference<decltype(u[0])>::type;
  if constexpr (VecLen == 1) {
  } else if constexpr (VecLen == 2) {
    Vec v[2];
    v[0] = blend2<0,2>(u[0],u[1]);
    v[1] = blend2<1,3>(u[0],u[1]);
    u[0] = v[0];
    u[1] = v[1];
  } else if constexpr (VecLen == 4) {
    Vec v[4];
    v[0] = blend4<0,4,2,6>(u[0],u[1]);
    v[1] = blend4<1,5,3,7>(u[0],u[1]);
    v[2] = blend4<0,4,2,6>(u[2],u[3]);
    v[3] = blend4<1,5,3,7>(u[2],u[3]);

    u[0] = blend4<0,1,4,5>(v[0],v[2]);
    u[1] = blend4<0,1,4,5>(v[1],v[3]);
    u[2] = blend4<2,3,6,7>(v[0],v[2]);
    u[3] = blend4<2,3,6,7>(v[1],v[3]);
  } else if constexpr (VecLen == 8) {
    Vec v[8], w[8];
    v[0] = blend8<0,8,2,10,4,12,6,14>(u[0],u[1]);
    v[1] = blend8<1,9,3,11,5,13,7,15>(u[0],u[1]);
    v[2] = blend8<0,8,2,10,4,12,6,14>(u[2],u[3]);
    v[3] = blend8<1,9,3,11,5,13,7,15>(u[2],u[3]);
    v[4] = blend8<0,8,2,10,4,12,6,14>(u[4],u[5]);
    v[5] = blend8<1,9,3,11,5,13,7,15>(u[4],u[5]);
    v[6] = blend8<0,8,2,10,4,12,6,14>(u[6],u[7]);
    v[7] = blend8<1,9,3,11,5,13,7,15>(u[6],u[7]);

    w[0] = blend8<0,1, 8, 9,4,5,12,13>(v[0],v[2]);
    w[1] = blend8<0,1, 8, 9,4,5,12,13>(v[1],v[3]);
    w[2] = blend8<2,3,10,11,6,7,14,15>(v[0],v[2]);
    w[3] = blend8<2,3,10,11,6,7,14,15>(v[1],v[3]);
    w[4] = blend8<0,1, 8, 9,4,5,12,13>(v[4],v[6]);
    w[5] = blend8<0,1, 8, 9,4,5,12,13>(v[5],v[7]);
    w[6] = blend8<2,3,10,11,6,7,14,15>(v[4],v[6]);
    w[7] = blend8<2,3,10,11,6,7,14,15>(v[5],v[7]);

    u[0] = blend8<0,1,2,3, 8, 9,10,11>(w[0],w[4]);
    u[1] = blend8<0,1,2,3, 8, 9,10,11>(w[1],w[5]);
    u[2] = blend8<0,1,2,3, 8, 9,10,11>(w[2],w[6]);
    u[3] = blend8<0,1,2,3, 8, 9,10,11>(w[3],w[7]);
    u[4] = blend8<4,5,6,7,12,13,14,15>(w[0],w[4]);
    u[5] = blend8<4,5,6,7,12,13,14,15>(w[1],w[5]);
    u[6] = blend8<4,5,6,7,12,13,14,15>(w[2],w[6]);
    u[7] = blend8<4,5,6,7,12,13,14,15>(w[3],w[7]);
  } else if constexpr (VecLen == 16) {
    Vec v[16], w[16];
    v[ 0] = blend16<0,16,2,18,4,20,6,22,8,24,10,26,12,28,14,30>(u[ 0],u[ 1]);
    v[ 1] = blend16<1,17,3,19,5,21,7,23,9,25,11,27,13,29,15,31>(u[ 0],u[ 1]);
    v[ 2] = blend16<0,16,2,18,4,20,6,22,8,24,10,26,12,28,14,30>(u[ 2],u[ 3]);
    v[ 3] = blend16<1,17,3,19,5,21,7,23,9,25,11,27,13,29,15,31>(u[ 2],u[ 3]);
    v[ 4] = blend16<0,16,2,18,4,20,6,22,8,24,10,26,12,28,14,30>(u[ 4],u[ 5]);
    v[ 5] = blend16<1,17,3,19,5,21,7,23,9,25,11,27,13,29,15,31>(u[ 4],u[ 5]);
    v[ 6] = blend16<0,16,2,18,4,20,6,22,8,24,10,26,12,28,14,30>(u[ 6],u[ 7]);
    v[ 7] = blend16<1,17,3,19,5,21,7,23,9,25,11,27,13,29,15,31>(u[ 6],u[ 7]);
    v[ 8] = blend16<0,16,2,18,4,20,6,22,8,24,10,26,12,28,14,30>(u[ 8],u[ 9]);
    v[ 9] = blend16<1,17,3,19,5,21,7,23,9,25,11,27,13,29,15,31>(u[ 8],u[ 9]);
    v[10] = blend16<0,16,2,18,4,20,6,22,8,24,10,26,12,28,14,30>(u[10],u[11]);
    v[11] = blend16<1,17,3,19,5,21,7,23,9,25,11,27,13,29,15,31>(u[10],u[11]);
    v[12] = blend16<0,16,2,18,4,20,6,22,8,24,10,26,12,28,14,30>(u[12],u[13]);
    v[13] = blend16<1,17,3,19,5,21,7,23,9,25,11,27,13,29,15,31>(u[12],u[13]);
    v[14] = blend16<0,16,2,18,4,20,6,22,8,24,10,26,12,28,14,30>(u[14],u[15]);
    v[15] = blend16<1,17,3,19,5,21,7,23,9,25,11,27,13,29,15,31>(u[14],u[15]);

    w[ 0] = blend16<0,1,16,17,4,5,20,21, 8, 9,24,25,12,13,28,29>(v[ 0],v[ 2]);
    w[ 1] = blend16<0,1,16,17,4,5,20,21, 8, 9,24,25,12,13,28,29>(v[ 1],v[ 3]);
    w[ 2] = blend16<2,3,18,19,6,7,22,23,10,11,26,27,14,15,30,31>(v[ 0],v[ 2]);
    w[ 3] = blend16<2,3,18,19,6,7,22,23,10,11,26,27,14,15,30,31>(v[ 1],v[ 3]);
    w[ 4] = blend16<0,1,16,17,4,5,20,21, 8, 9,24,25,12,13,28,29>(v[ 4],v[ 6]);
    w[ 5] = blend16<0,1,16,17,4,5,20,21, 8, 9,24,25,12,13,28,29>(v[ 5],v[ 7]);
    w[ 6] = blend16<2,3,18,19,6,7,22,23,10,11,26,27,14,15,30,31>(v[ 4],v[ 6]);
    w[ 7] = blend16<2,3,18,19,6,7,22,23,10,11,26,27,14,15,30,31>(v[ 5],v[ 7]);
    w[ 8] = blend16<0,1,16,17,4,5,20,21, 8, 9,24,25,12,13,28,29>(v[ 8],v[10]);
    w[ 9] = blend16<0,1,16,17,4,5,20,21, 8, 9,24,25,12,13,28,29>(v[ 9],v[11]);
    w[10] = blend16<2,3,18,19,6,7,22,23,10,11,26,27,14,15,30,31>(v[ 8],v[10]);
    w[11] = blend16<2,3,18,19,6,7,22,23,10,11,26,27,14,15,30,31>(v[ 9],v[11]);
    w[12] = blend16<0,1,16,17,4,5,20,21, 8, 9,24,25,12,13,28,29>(v[12],v[14]);
    w[13] = blend16<0,1,16,17,4,5,20,21, 8, 9,24,25,12,13,28,29>(v[13],v[15]);
    w[14] = blend16<2,3,18,19,6,7,22,23,10,11,26,27,14,15,30,31>(v[12],v[14]);
    w[15] = blend16<2,3,18,19,6,7,22,23,10,11,26,27,14,15,30,31>(v[13],v[15]);

    v[ 0] = blend16<0,1,2,3, 16,17,18,19,  8, 9,10,11, 24,25,26,27>(w[ 0],w[ 4]);
    v[ 1] = blend16<0,1,2,3, 16,17,18,19,  8, 9,10,11, 24,25,26,27>(w[ 1],w[ 5]);
    v[ 2] = blend16<0,1,2,3, 16,17,18,19,  8, 9,10,11, 24,25,26,27>(w[ 2],w[ 6]);
    v[ 3] = blend16<0,1,2,3, 16,17,18,19,  8, 9,10,11, 24,25,26,27>(w[ 3],w[ 7]);
    v[ 4] = blend16<4,5,6,7, 20,21,22,23, 12,13,14,15, 28,29,30,31>(w[ 0],w[ 4]);
    v[ 5] = blend16<4,5,6,7, 20,21,22,23, 12,13,14,15, 28,29,30,31>(w[ 1],w[ 5]);
    v[ 6] = blend16<4,5,6,7, 20,21,22,23, 12,13,14,15, 28,29,30,31>(w[ 2],w[ 6]);
    v[ 7] = blend16<4,5,6,7, 20,21,22,23, 12,13,14,15, 28,29,30,31>(w[ 3],w[ 7]);
    v[ 8] = blend16<0,1,2,3, 16,17,18,19,  8, 9,10,11, 24,25,26,27>(w[ 8],w[12]);
    v[ 9] = blend16<0,1,2,3, 16,17,18,19,  8, 9,10,11, 24,25,26,27>(w[ 9],w[13]);
    v[10] = blend16<0,1,2,3, 16,17,18,19,  8, 9,10,11, 24,25,26,27>(w[10],w[14]);
    v[11] = blend16<0,1,2,3, 16,17,18,19,  8, 9,10,11, 24,25,26,27>(w[11],w[15]);
    v[12] = blend16<4,5,6,7, 20,21,22,23, 12,13,14,15, 28,29,30,31>(w[ 8],w[12]);
    v[13] = blend16<4,5,6,7, 20,21,22,23, 12,13,14,15, 28,29,30,31>(w[ 9],w[13]);
    v[14] = blend16<4,5,6,7, 20,21,22,23, 12,13,14,15, 28,29,30,31>(w[10],w[14]);
    v[15] = blend16<4,5,6,7, 20,21,22,23, 12,13,14,15, 28,29,30,31>(w[11],w[15]);

    u[ 0] = blend16<0,1,2,3,4,5,6,7, 16,17,18,19,20,21,22,23>(v[0],v[ 8]);
    u[ 1] = blend16<0,1,2,3,4,5,6,7, 16,17,18,19,20,21,22,23>(v[1],v[ 9]);
    u[ 2] = blend16<0,1,2,3,4,5,6,7, 16,17,18,19,20,21,22,23>(v[2],v[10]);
    u[ 3] = blend16<0,1,2,3,4,5,6,7, 16,17,18,19,20,21,22,23>(v[3],v[11]);
    u[ 4] = blend16<0,1,2,3,4,5,6,7, 16,17,18,19,20,21,22,23>(v[4],v[12]);
    u[ 5] = blend16<0,1,2,3,4,5,6,7, 16,17,18,19,20,21,22,23>(v[5],v[13]);
    u[ 6] = blend16<0,1,2,3,4,5,6,7, 16,17,18,19,20,21,22,23>(v[6],v[14]);
    u[ 7] = blend16<0,1,2,3,4,5,6,7, 16,17,18,19,20,21,22,23>(v[7],v[15]);
    u[ 8] = blend16<8,9,10,11,12,13,14,15, 24,25,26,27,28,29,30,31>(v[0],v[ 8]);
    u[ 9] = blend16<8,9,10,11,12,13,14,15, 24,25,26,27,28,29,30,31>(v[1],v[ 9]);
    u[10] = blend16<8,9,10,11,12,13,14,15, 24,25,26,27,28,29,30,31>(v[2],v[10]);
    u[11] = blend16<8,9,10,11,12,13,14,15, 24,25,26,27,28,29,30,31>(v[3],v[11]);
    u[12] = blend16<8,9,10,11,12,13,14,15, 24,25,26,27,28,29,30,31>(v[4],v[12]);
    u[13] = blend16<8,9,10,11,12,13,14,15, 24,25,26,27,28,29,30,31>(v[5],v[13]);
    u[14] = blend16<8,9,10,11,12,13,14,15, 24,25,26,27,28,29,30,31>(v[6],v[14]);
    u[15] = blend16<8,9,10,11,12,13,14,15, 24,25,26,27,28,29,30,31>(v[7],v[15]);
  } else {
    static_assert(VecLen!=VecLen, "not implemented");
  }
}

template <long q> constexpr long MatInvFLOP_count() {
  if constexpr (q==3) return 42;
  else if constexpr (q==4) return 158;
  else static_assert(q != q, "Not implemented");
  return 0;
}

template <long q, class Real> static void MatInv_naive(Real* Minv, const Real* M, long N) {
  for (long i = 0; i < N; i++) {
    const Real* M_ = M + i*q*q;
    Real* Minv_ = Minv + i*q*q;
    MatInvKernel<q>(Minv_, M_);
  }
}

template <long q, long VecLen, class Real> static void MatInv_vec(Real* Minv, const Real* M, long N) {
  using Vec = Vec<Real, VecLen>;
  const long remaining = N % VecLen;
  const long N_ = N - remaining;

  for (long i = 0; i < N_; i += VecLen) {
    const Real* M_ = M + i*q*q;
    Real* Minv_ = Minv + i*q*q;

    constexpr long Nblocks = q*q/VecLen;
    //constexpr long Nblocks = (q*q+VecLen-1)/VecLen; // input and output arrays must be padded
    constexpr long qq = std::max<long>(q*q, Nblocks*VecLen);

    Vec M[qq], Minv[qq];
    for (long i = 0; i < Nblocks; i++) { // vector loads followed by transpose
      auto M__ = M + i*VecLen;
      for (long k = 0; k < VecLen; k++) {
        M__[k].load(M_ + i*VecLen + k*q*q);
      }
      VecTranspose<VecLen>(M__);
    }
    for (long k = Nblocks*VecLen; k < q*q; k++) { // scalar load for remaining elements
      if constexpr (VecLen ==  2) M[k] = Vec(M_[k],M_[k+1*q*q]);
      if constexpr (VecLen ==  4) M[k] = Vec(M_[k],M_[k+1*q*q],M_[k+2*q*q],M_[k+3*q*q]);
      if constexpr (VecLen ==  8) M[k] = Vec(M_[k],M_[k+1*q*q],M_[k+2*q*q],M_[k+3*q*q],M_[k+4*q*q],M_[k+5*q*q],M_[k+6*q*q],M_[k+7*q*q]);
      if constexpr (VecLen == 16) M[k] = Vec(M_[k],M_[k+1*q*q],M_[k+2*q*q],M_[k+3*q*q],M_[k+4*q*q],M_[k+5*q*q],M_[k+6*q*q],M_[k+7*q*q],M_[k+8*q*q],M_[k+9*q*q],M_[k+10*q*q],M_[k+11*q*q],M_[k+12*q*q],M_[k+13*q*q],M_[k+14*q*q],M_[k+15*q*q]);
    }

    MatInvKernel<q>(Minv, M);

    for (long i = 0; i < Nblocks; i++) { // transpose
      auto Minv__ = Minv + i*VecLen;
      VecTranspose<VecLen>(Minv__);
    }
    for (long k = 0; k < VecLen; k++) { // vector stores
      for (long i = 0; i < Nblocks; i++) {
        Minv[i*VecLen+k].store(Minv_ + k*q*q + i*VecLen);
      }
    }
    for (long k = Nblocks*VecLen; k < q*q; k++) { // scalar stores
      if constexpr (VecLen ==  2) {
        Minv_[k] = Minv[k][0];
        Minv_[k+1*q*q] = Minv[k][1];
      }
      if constexpr (VecLen ==  4) {
        Minv_[k] = Minv[k][0];
        Minv_[k+1*q*q] = Minv[k][1];
        Minv_[k+2*q*q] = Minv[k][2];
        Minv_[k+3*q*q] = Minv[k][3];
      }
      if constexpr (VecLen ==  8) {
        Minv_[k] = Minv[k][0];
        Minv_[k+1*q*q] = Minv[k][1];
        Minv_[k+2*q*q] = Minv[k][2];
        Minv_[k+3*q*q] = Minv[k][3];
        Minv_[k+4*q*q] = Minv[k][4];
        Minv_[k+5*q*q] = Minv[k][5];
        Minv_[k+6*q*q] = Minv[k][6];
        Minv_[k+7*q*q] = Minv[k][7];
      }
      if constexpr (VecLen == 16) {
        Minv_[k] = Minv[k][0];
        Minv_[k+1*q*q] = Minv[k][1];
        Minv_[k+2*q*q] = Minv[k][2];
        Minv_[k+3*q*q] = Minv[k][3];
        Minv_[k+4*q*q] = Minv[k][4];
        Minv_[k+5*q*q] = Minv[k][5];
        Minv_[k+6*q*q] = Minv[k][6];
        Minv_[k+7*q*q] = Minv[k][7];
        Minv_[k+8*q*q] = Minv[k][8];
        Minv_[k+9*q*q] = Minv[k][9];
        Minv_[k+10*q*q] = Minv[k][10];
        Minv_[k+11*q*q] = Minv[k][11];
        Minv_[k+12*q*q] = Minv[k][12];
        Minv_[k+13*q*q] = Minv[k][13];
        Minv_[k+14*q*q] = Minv[k][14];
        Minv_[k+15*q*q] = Minv[k][15];
      }
    }
  }

  MatInv_naive<q>(Minv + N_*q*q, M + N_*q*q, remaining);
}

constexpr long PackedSize(long N, long VecLen, long qq) {
  return ((N / qq + VecLen - 1) / VecLen) * VecLen * qq;
}

template <long qq, long VecLen, class Real> static void VecPack(Real* Y, long Ny, const Real* X, long Nx) {
  assert(Ny == PackedSize(Nx, VecLen, qq));
  const long N = (Nx / qq);
  const long NN = N / VecLen;
  for (long i = 0; i < NN; i++) {
    for (long j = 0; j < qq; j++) {
      for (long k = 0; k < VecLen; k++) {
        Y[(i*qq+j)*VecLen+k] = X[(i*VecLen+k)*qq+j];
      }
    }
  }
  for (long j = 0; j < qq; j++) {
    for (long k = 0; k < N % VecLen; k++) {
      Y[(NN*qq+j)*VecLen+k] = X[(NN*VecLen+k)*qq+j];
    }
  }
}

template <long q, long VecLen, class Real> static void MatInv_packed(Real* Minv, const Real* M, long N) {
  using Vec = Vec<Real, VecLen>;
  assert(N % VecLen == 0);

  for (long i = 0; i < N; i += VecLen) {
    const Real* M_ = M + i*q*q;
    Real* Minv_ = Minv + i*q*q;

    Vec M[q*q], Minv[q*q];
    for (long k = 0; k < q*q; k++) M[k].load(M_ + k*VecLen);
    MatInvKernel<q>(Minv, M);
    for (long k = 0; k < q*q; k++) Minv[k].store(Minv_ + k*VecLen);
  }
}

template <long qq, long VecLen, class Real> static void VecUnpack(Real* X, long Nx, const Real* Y, long Ny) {
  assert(Ny == PackedSize(Nx, VecLen, qq));
  const long N = (Nx / qq);
  const long NN = N / VecLen;
  for (long i = 0; i < NN; i++) {
    for (long j = 0; j < qq; j++) {
      for (long k = 0; k < VecLen; k++) {
        X[(i*VecLen+k)*qq+j] = Y[(i*qq+j)*VecLen+k];
      }
    }
  }
  for (long j = 0; j < qq; j++) {
    for (long k = 0; k < N % VecLen; k++) {
      X[(NN*VecLen+k)*qq+j] = Y[(NN*qq+j)*VecLen+k];
    }
  }
}

