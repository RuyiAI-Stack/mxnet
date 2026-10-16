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
 * \file reduce_with_axis.h
 * \brief
 * \author Junyuan Xie
*/
#ifndef MSHADOW_EXTENSION_REDUCE_WITH_AXIS_H_
#define MSHADOW_EXTENSION_REDUCE_WITH_AXIS_H_

#include "../extension.h"

namespace mshadow {
namespace expr {

/*!
 * \brief Accumulation step for the axis reducer.
 *
 *  `red::sum::Reduce` takes its accumulator as `volatile DType&`, which forces
 *  the running sum through memory on every reduction step.  For the additive
 *  reducer a plain register accumulation performs the same `dst += src`
 *  sequence, in the same order, without the volatile round-trip.  Every other
 *  reducer keeps the existing `Reducer::Reduce` contract.
 *
 *  \note Dropping the `volatile` qualification allows the compiler to contract
 *  `acc + val` into a fused multiply-add whenever `val` is itself a product,
 *  as happens for `sum(lhs * rhs)`; the last bits of the result may therefore
 *  differ from the volatile version.  Callers that need bit-identical output
 *  across compilers should be aware of this.
 */
template<typename DType>
MSHADOW_XINLINE void ReduceWithAxisStep(red::sum, DType &acc, DType val) {  // NOLINT(*)
  acc += val;
}
template<typename Reducer, typename DType>
MSHADOW_XINLINE void ReduceWithAxisStep(Reducer, DType &acc, DType val) {  // NOLINT(*)
  Reducer::Reduce(acc, val);
}

/*! \brief reduce out the dimension of src labeled by axis.
 *  \tparam Reducer type of reducer
 *  \tparam SrcExp type of source expression
 *  \tparam DType data type
 */
template<typename Reducer, typename SrcExp, typename DType, int dimsrc, bool mask, int dimdst>
struct ReduceWithAxisExp:
    public MakeTensorExp<ReduceWithAxisExp<Reducer, SrcExp, DType, dimsrc, mask, dimdst>,
                         SrcExp, dimdst, DType> {
  /*! \brief source oprand */
  const SrcExp &src_;
  /*! \brief size of last destination dimension */
  index_t last_dst_dim_;
  /*! \brief size of trailing dimensions */
  index_t trailing_;
  /*! \brief size of axis dimension */
  index_t size_;
  /*! \brief size of last src dimension */
  index_t last_;
  /*! constructor */
  explicit ReduceWithAxisExp(const SrcExp &src, int axis)
    : src_(src) {
    bool keepdim = (dimsrc == dimdst);
    CHECK(dimsrc > axis) << "reduce axis out of bound";
    Shape<dimsrc> src_shape = ShapeCheck<dimsrc, SrcExp>::Check(src_);
    for (int i = 0; i < axis; ++i) {
      this->shape_[i] = src_shape[i];
    }
    this->size_ = src_shape[axis];
    this->trailing_ = 1;
    if (!keepdim) {
      for (int i = axis + 1; i < dimsrc; ++i) {
        this->trailing_ *= src_shape[i];
        this->shape_[i - 1] = src_shape[i];
      }
    } else {
      this->shape_[axis] = 1;
      for (index_t i = axis + 1; i < dimsrc; ++i) {
        this->trailing_ *= src_shape[i];
        this->shape_[i] = src_shape[i];
      }
    }

    this->last_ = src_shape[dimsrc - 1];
    this->last_dst_dim_ = this->shape_[dimdst - 1];
  }
};  // struct ReduceWithAxisExp

/*!
 * \brief reduce out the dimension of src labeled by axis.
 * \param Reducer type of the reducing operation
 * \param mask whether to output the unmask indices
 * \tparam SrcExp source expression
 * \tparam DType data type
 * \tparam etype type of the expression
 */
template<typename Reducer, bool mask, typename SrcExp, typename DType, int etype>
inline ReduceWithAxisExp<Reducer, SrcExp, DType, ExpInfo<SrcExp>::kDim, mask,
  ExpInfo<SrcExp>::kDim - 1>
reduce_with_axis(const Exp<SrcExp, DType, etype> &src, int axis) {
  return ReduceWithAxisExp<Reducer, SrcExp, DType, ExpInfo<SrcExp>::kDim, mask,
    ExpInfo<SrcExp>::kDim- 1>(src.self(), axis);
}

/*!
* \brief reduce out the dimension of src labeled by axis, keepdim turned on.
* \param Reducer type of the reducing operation
* \param mask whether to output the unmask indices
* \tparam SrcExp source expression
* \tparam DType data type
* \tparam etype type of the expression
*/
template<typename Reducer, bool mask, typename SrcExp, typename DType, int etype>
inline ReduceWithAxisExp<Reducer, SrcExp, DType, ExpInfo<SrcExp>::kDim, mask,
  ExpInfo<SrcExp>::kDim>
  reduce_keepdim(const Exp<SrcExp, DType, etype> &src, int axis) {
  return ReduceWithAxisExp<Reducer, SrcExp, DType, ExpInfo<SrcExp>::kDim, mask,
    ExpInfo<SrcExp>::kDim>(src.self(), axis);
}

//----------------------
// Execution plan
//----------------------
template<typename Reducer, typename SrcExp, typename DType, int dimsrc, bool mask, int dimdst>
struct Plan<ReduceWithAxisExp<Reducer, SrcExp, DType, dimsrc, mask, dimdst>, DType> {
 public:
  explicit Plan(const ReduceWithAxisExp<Reducer, SrcExp, DType, dimsrc, mask, dimdst> &e)
      : src_(MakePlan(e.src_)), last_dst_dim_(e.last_dst_dim_), trailing_(e.trailing_),
        size_(e.size_), last_(e.last_) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    index_t x = (i*last_dst_dim_ + j)/trailing_;
    index_t y = (i*last_dst_dim_ + j)%trailing_;

    // Empty reduction: keep the original early-exit semantics (init value for
    // the value path, 0 for the masked index path) and avoid the pre-loop
    // decode division when last_ == 0.
    if (size_ == 0) {
      if (mask) return static_cast<DType>(0);
      DType empty_res; Reducer::SetInitValue(empty_res);
      return empty_res;
    }

    // Strength-reduced index decode: z = base_z + k*trailing_ with
    // base_z = x*size_*trailing_ + y, so z/last_ and z%last_ advance by the
    // constant (q, r) = (trailing_/last_, trailing_%last_) plus a single
    // carry.  Integer-equivalent to the original per-element z/last_, z%last_
    // decode, but with no division inside the k loop.
    const index_t q = trailing_ / last_;
    const index_t r = trailing_ % last_;
    index_t zdiv = (x*size_*trailing_ + y) / last_;
    index_t zmod = (x*size_*trailing_ + y) % last_;

    if (mask) {
      index_t idx = 0;
      DType res; Reducer::SetInitValue(res);
      for (index_t k = 0; k < size_; ++k) {
        DType tmp = res;
        ReduceWithAxisStep(Reducer(), res, src_.Eval(zdiv, zmod));
        if (tmp != res && !isnan_typed::IsNan(tmp)) {
          idx = k;
        }
        zdiv += q; zmod += r;
        if (zmod >= last_) { zmod -= last_; ++zdiv; }
      }
      return static_cast<DType>(static_cast<int>(idx));
    } else {
      DType res; Reducer::SetInitValue(res);
      for (index_t k = 0; k < size_; ++k) {
        ReduceWithAxisStep(Reducer(), res, src_.Eval(zdiv, zmod));
        zdiv += q; zmod += r;
        if (zmod >= last_) { zmod -= last_; ++zdiv; }
      }
      return res;
    }
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t last_dst_dim_, trailing_, size_, last_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_REDUCE_WITH_AXIS_H_
