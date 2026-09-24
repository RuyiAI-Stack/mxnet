/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file spatial_upsampling.h
 * \brief
 * \author Bing Xu
*/
#ifndef MSHADOW_EXTENSION_SPATIAL_UPSAMPLING_NEAREST_H_
#define MSHADOW_EXTENSION_SPATIAL_UPSAMPLING_NEAREST_H_
#include <cstdint>
#include <vector>
#include "../extension.h"
#if defined(__riscv_v_intrinsic) || defined(__riscv_vector)
#include <riscv_vector.h>
#endif

namespace mshadow {
namespace expr {

/*! \brief nearest neighboor upsampling
 *         out(x, y) = in(int(x / scale_x), int(y / scale_y))
 *  \tparam SrcExp source expression
 *  \tparam DType data type
 *  \tparam srcdim source dimension
 */
template<typename SrcExp, typename DType, int srcdim>
struct UpSamplingNearestExp :
  public MakeTensorExp<UpSamplingNearestExp<SrcExp, DType, srcdim>,
                       SrcExp, srcdim, DType> {
  /*! \brief source oprand */
  const SrcExp &src_;
  /*! \brief up sampling scale */
  index_t scale_;
  /*! \brief constructor */
  UpSamplingNearestExp(const SrcExp &src, index_t scale)
    : src_(src), scale_(scale) {
    this->shape_ = ShapeCheck<srcdim, SrcExp>::Check(src_);
    this->shape_[srcdim - 2] *= scale_;
    this->shape_[srcdim - 1] *= scale_;
  }
};


template<typename SrcExp, typename DType, int etype>
inline UpSamplingNearestExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
upsampling_nearest(const Exp<SrcExp, DType, etype> &src, index_t scale) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim >= 2>
    ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return UpSamplingNearestExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>(src.self(), scale);
}

template<typename SrcExp, typename DType, int srcdim>
struct Plan<UpSamplingNearestExp<SrcExp, DType, srcdim>, DType> {
 public:
  explicit Plan(const UpSamplingNearestExp<SrcExp, DType, srcdim> &e)
    : src_(MakePlan(e.src_)),
      scale_(e.scale_),
      new_height_(e.shape_[srcdim - 2]),
      src_height_(static_cast<index_t>(e.shape_[srcdim - 2] / e.scale_)) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    const index_t x = j;
    const index_t y = i % new_height_;
    const index_t c = i / new_height_;
    const index_t h = static_cast<index_t>(y / scale_);
    const index_t w = static_cast<index_t>(x / scale_);
    return src_.Eval(c * src_height_ + h, w);
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t scale_;
  const index_t new_height_;
  const index_t src_height_;
};

#if defined(__riscv_v_intrinsic) || defined(__riscv_vector)
/*!
 * \brief RVV 1.0 nearest-neighbour upsampling for contiguous NCHW float32
 *        tensors with a single integer scale.
 *
 *        The horizontal source byte-offset table w_byte[x] = (x / scale) * 4
 *        depends only on `scale` and the output width, so it is built once per
 *        call and reused for every output row and channel, removing the
 *        per-element integer division of the scalar reference. The main loop
 *        then performs a vluxei32 indexed load from the current source row and
 *        a contiguous vse32 store; runtime vsetvl keeps the code
 *        VLEN-agnostic. floor(x / scale) / floor(y / scale) semantics match
 *        Plan<UpSamplingNearestExp>::Eval element-for-element.
 *
 *        Preconditions (checked by the caller): src/dst contiguous and
 *        non-overlapping, dst_height == src_height * scale,
 *        dst_width == src_width * scale, scale >= 1 and
 *        (src_width - 1) * sizeof(float) fits in uint32_t.
 */
inline void mshadow_rvv_upsampling_nearest_f32(const float *src, float *dst,
                                               index_t planes,
                                               index_t src_height,
                                               index_t src_width,
                                               index_t scale) {
  if (planes <= 0 || src_height <= 0 || src_width <= 0 || scale <= 0) return;
  const index_t dst_height = src_height * scale;
  const index_t dst_width = src_width * scale;
  if (scale == 2) {
    const index_t total_rows = planes * dst_height;
    const size_t vlmax = __riscv_vsetvlmax_e32m4();
#ifndef __CUDACC__
    #pragma omp parallel for
#endif
    for (index_t r = 0; r < total_rows; ++r) {
      const index_t p = r / dst_height;
      const index_t y = r % dst_height;
      const float *src_row = src + (p * src_height + y / scale) * src_width;
      float *dst_row = dst + (p * dst_height + y) * dst_width;
      index_t x = 0;
      while (x < src_width) {
        const size_t remaining = static_cast<size_t>(src_width - x);
        const size_t src_count = remaining < vlmax / 2 ? remaining : vlmax / 2;
        const size_t vl_src = __riscv_vsetvl_e32m4(src_count);
        const vfloat32m4_t vin = __riscv_vle32_v_f32m4(src_row + x, vl_src);
        const size_t vl_out = __riscv_vsetvl_e32m4(vl_src * 2);
        vuint32m4_t index = __riscv_vid_v_u32m4(vl_out);
        index = __riscv_vsrl_vx_u32m4(index, 1, vl_out);
        const vfloat32m4_t vout = __riscv_vrgather_vv_f32m4(vin, index, vl_out);
        __riscv_vse32_v_f32m4(dst_row + x * 2, vout, vl_out);
        x += static_cast<index_t>(vl_src);
      }
    }
    return;
  }
  const index_t float_bytes = static_cast<index_t>(sizeof(float));
  std::vector<uint32_t> w_byte(static_cast<size_t>(dst_width));
  for (index_t x = 0, sw = 0, in_group = 0; x < dst_width; ++x) {
    w_byte[static_cast<size_t>(x)] = static_cast<uint32_t>(sw * float_bytes);
    if (++in_group == scale) {
      in_group = 0;
      ++sw;
    }
  }
  // Row-level parallelism, matching the original MapPlan `omp parallel for`
  // over flattened rows; the offset table is read-only and shared.
  const index_t total_rows = planes * dst_height;
#ifndef __CUDACC__
  #pragma omp parallel for
#endif
  for (index_t r = 0; r < total_rows; ++r) {
    const index_t p = r / dst_height;
    const index_t y = r % dst_height;
    const float *src_row = src + (p * src_height + y / scale) * src_width;
    float *dst_row = dst + (p * dst_height + y) * dst_width;
    index_t x = 0;
    while (x < dst_width) {
      size_t vl = __riscv_vsetvl_e32m4(static_cast<size_t>(dst_width - x));
      vuint32m4_t voff = __riscv_vle32_v_u32m4(w_byte.data() + x, vl);
      vfloat32m4_t v = __riscv_vluxei32_v_f32m4(src_row, voff, vl);
      __riscv_vse32_v_f32m4(dst_row + x, v, vl);
      x += static_cast<index_t>(vl);
    }
  }
}
#endif  // defined(__riscv_v_intrinsic) || defined(__riscv_vector)

}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_SPATIAL_UPSAMPLING_NEAREST_H_
