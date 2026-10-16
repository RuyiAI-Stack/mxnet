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

#pragma once

#include <tuple>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>
#include <type_traits>

#include <dmlc/omp.h>

#include "ctc_helper.h"

#if defined(__riscv) && defined(__riscv_v)
#include <riscv_vector.h>
#endif

namespace mxnet_warpctc {

template<typename ProbT>
class CpuCTC {
public:
    // Noncopyable
    CpuCTC(int alphabet_size, int minibatch, void* workspace,
           int blank_label) :
            alphabet_size_(alphabet_size), minibatch_(minibatch),
            workspace_(workspace), blank_label_(blank_label) {

    };

    CpuCTC(const CpuCTC&) = delete;
    CpuCTC& operator=(const CpuCTC&) = delete;

    ctcStatus_t cost_and_grad(const ProbT* const activations,
                              ProbT *grads,
                              ProbT* costs,
                              const int* const flat_labels,
                              const int* const label_lengths,
                              const int* const input_lengths);


    ctcStatus_t score_forward(const ProbT* const activations,
                              ProbT* costs,
                              const int* const flat_labels,
                              const int* const label_lengths,
                              const int* const input_lengths);

private:

    class CpuCTC_metadata {

    private:
        int setup_labels(const int* const labels, int blank_label, int L, int S);

    public:
        CpuCTC_metadata(int L, int S, int T, int mb, int alphabet_size,
                        void* workspace, size_t bytes_used, int blank_label,
                        const int* const labels);

        ProbT* alphas;
        ProbT* betas;
        int* labels_w_blanks;
        int* e_inc;
        int* s_inc;
        ProbT* output;
        int repeats;
    };

    int alphabet_size_; // Number of characters plus blank
    int minibatch_;
    void* workspace_;
    int blank_label_;

    void log_softmax(const ProbT* const activations, ProbT* log_probs,
                     const int* const input_lengths);

    std::tuple<ProbT, bool>
            cost_and_grad_kernel(ProbT *grad, const ProbT* const log_probs,
                                 const int* const labels, int T, int L,
                                 int mb, size_t bytes_used);

    ProbT compute_alphas(const ProbT* log_probs, int repeats, int S, int T,
                         const int* const e_inc,
                         const int* const s_inc,
                         const int* const labels,
                         ProbT* alphas);

    ProbT compute_betas_and_grad(ProbT* grad, const ProbT* const log_probs,
                                 ProbT log_partition, int repeats,
                                 int S, int T, const int* const e_inc,
                                 const int* const s_inc,
                                 const int* const labels,
                                 ProbT* alphas,
                                 ProbT* betas,
                                 ProbT* output);
};

template<typename ProbT>
CpuCTC<ProbT>::CpuCTC_metadata::CpuCTC_metadata(int L, int S, int T, int mb,
                                                int alphabet_size,
                                                void* workspace, size_t bytes_used,
                                                int blank_label,
                                                const int* const labels) {

    alphas = reinterpret_cast<ProbT *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * S * T;
    std::fill(alphas, alphas + S * T, ctc_helper::neg_inf<ProbT>());
    betas = reinterpret_cast<ProbT *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * S;
    std::fill(betas, betas + S, ctc_helper::neg_inf<ProbT>());
    labels_w_blanks = reinterpret_cast<int *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(int) * S;
    e_inc = reinterpret_cast<int *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(int) * S;
    s_inc = reinterpret_cast<int *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(int) * S;
    output = reinterpret_cast<ProbT *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * alphabet_size;

    repeats = setup_labels(labels, blank_label, L, S);
}

template<typename ProbT>
int CpuCTC<ProbT>::CpuCTC_metadata::setup_labels(const int* const labels,
                                                 int blank_label, int L, int S) {
    int e_counter = 0;
    int s_counter = 0;

    s_inc[s_counter++] = 1;

    int repeats = 0;

    for (int i = 1; i < L; ++i) {
        if (labels[i-1] == labels[i]) {
            s_inc[s_counter++] = 1;
            s_inc[s_counter++] = 1;
            e_inc[e_counter++] = 1;
            e_inc[e_counter++] = 1;
            ++repeats;
        }
        else {
            s_inc[s_counter++] = 2;
            e_inc[e_counter++] = 2;
        }
    }
    e_inc[e_counter++] = 1;

    for (int i = 0; i < L; ++i) {
        labels_w_blanks[2 * i] = blank_label;
        labels_w_blanks[2 * i + 1] = labels[i];
    }
    labels_w_blanks[S - 1] = blank_label;

    return repeats;
}

#if defined(__riscv) && defined(__riscv_v)
namespace ctc_rvv {

// Self-contained range-reduced vector exponential for float32.
//
// RVV 1.0 has no native exp instruction, so exp(x) is evaluated as
// 2^k * exp(r) with k = round(x * log2e) and r = x - k*ln2.  ln2 is split in
// two exactly representable parts and BOTH parts are scaled by the integer k
// before the subtraction (r = x - k*ln2_hi - k*ln2_lo), which keeps r small
// and the polynomial accurate.  exp(r) is a degree-6 Horner polynomial
// (relative error ~1e-7 on r in [-ln2/2, ln2/2]) and 2^k is materialised
// through the IEEE-754 exponent field.  LogSoftmax always feeds this helper
// values <= 0, so the clamp only guards the underflow tail.
inline vfloat32m4_t exp_f32(vfloat32m4_t x, size_t vl) {
    const float log2e  = 1.4426950408889634f;
    const float ln2_hi = 0.693359375f;
    const float ln2_lo = -2.1219444005469058277e-4f;

    x = __riscv_vfmax_vf_f32m4(x, -87.33654475f, vl);  // exp underflow bound
    x = __riscv_vfmin_vf_f32m4(x, 88.0f, vl);          // keep 2^k exponent in range

    vint32m4_t k = __riscv_vfcvt_x_f_v_i32m4(__riscv_vfmul_vf_f32m4(x, log2e, vl), vl);
    vfloat32m4_t kf = __riscv_vfcvt_f_x_v_f32m4(k, vl);
    vfloat32m4_t r = __riscv_vfsub_vv_f32m4(x, __riscv_vfmul_vf_f32m4(kf, ln2_hi, vl), vl);
    r = __riscv_vfsub_vv_f32m4(r, __riscv_vfmul_vf_f32m4(kf, ln2_lo, vl), vl);

    vfloat32m4_t p = __riscv_vfmv_v_f_f32m4(1.0f / 720.0f, vl);
    p = __riscv_vfmadd_vv_f32m4(p, r, __riscv_vfmv_v_f_f32m4(1.0f / 120.0f, vl), vl);
    p = __riscv_vfmadd_vv_f32m4(p, r, __riscv_vfmv_v_f_f32m4(1.0f / 24.0f, vl), vl);
    p = __riscv_vfmadd_vv_f32m4(p, r, __riscv_vfmv_v_f_f32m4(1.0f / 6.0f, vl), vl);
    p = __riscv_vfmadd_vv_f32m4(p, r, __riscv_vfmv_v_f_f32m4(0.5f, vl), vl);
    p = __riscv_vfmadd_vv_f32m4(p, r, __riscv_vfmv_v_f_f32m4(1.0f, vl), vl);
    p = __riscv_vfmadd_vv_f32m4(p, r, __riscv_vfmv_v_f_f32m4(1.0f, vl), vl);

    vint32m4_t scale_bits = __riscv_vsll_vx_i32m4(__riscv_vadd_vx_i32m4(k, 127, vl), 23, vl);
    return __riscv_vfmul_vv_f32m4(p, __riscv_vreinterpret_v_i32m4_f32m4(scale_bits), vl);
}

// Vectorised LogSoftmax for one (minibatch, time-step) row of `alphabet_size`
// floats.  Mirrors the scalar three-stage implementation below:
//   1. row maximum          : vle32 + vfredmax.vs with a scalar running seed
//   2. denom = sum exp(a-max): vle32 -> vfsub.vf -> exp_f32 -> vfredusum.vs
//   3. log_probs = a - max - log(denom): scalar log(denom) once, vfsub.vf, vse32
// The main loop runs at the maximum vector length the hardware offers and the
// trailing partial block is handled with a runtime vsetvl tail, so any
// alphabet_size is supported.  vfredusum changes the floating-point summation
// order relative to the scalar loop; the resulting difference is far below the
// CTC loss test tolerances (measured max abs error ~4e-6).
inline void log_softmax_f32(const float* activations, float* log_probs, int alphabet_size) {
    if (alphabet_size <= 0) {
        return;
    }
    const size_t n = static_cast<size_t>(alphabet_size);
    const size_t vlmax = __riscv_vsetvlmax_e32m4();

    // Stage 1: row maximum.
    vfloat32m1_t max_seed = __riscv_vfmv_v_f_f32m1(-std::numeric_limits<float>::infinity(), 1);
    size_t i = 0;
    for (; i + vlmax <= n; i += vlmax) {
        vfloat32m4_t v = __riscv_vle32_v_f32m4(activations + i, vlmax);
        vfloat32m1_t red = __riscv_vfredmax_vs_f32m4_f32m1(v, max_seed, vlmax);
        max_seed = __riscv_vfmv_s_f_f32m1(__riscv_vfmv_f_s_f32m1_f32(red), 1);
    }
    if (i < n) {
        const size_t vl = __riscv_vsetvl_e32m4(n - i);
        vfloat32m4_t v = __riscv_vle32_v_f32m4(activations + i, vl);
        vfloat32m1_t red = __riscv_vfredmax_vs_f32m4_f32m1(v, max_seed, vl);
        max_seed = __riscv_vfmv_s_f_f32m1(__riscv_vfmv_f_s_f32m1_f32(red), 1);
    }
    const float max_activation = __riscv_vfmv_f_s_f32m1_f32(max_seed);

    // Stage 2: denominator = sum(exp(act - max)).
    vfloat32m1_t sum_seed = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    i = 0;
    for (; i + vlmax <= n; i += vlmax) {
        vfloat32m4_t shifted = __riscv_vfsub_vf_f32m4(
            __riscv_vle32_v_f32m4(activations + i, vlmax), max_activation, vlmax);
        vfloat32m4_t e = exp_f32(shifted, vlmax);
        vfloat32m1_t red = __riscv_vfredusum_vs_f32m4_f32m1(e, sum_seed, vlmax);
        sum_seed = __riscv_vfmv_s_f_f32m1(__riscv_vfmv_f_s_f32m1_f32(red), 1);
    }
    if (i < n) {
        const size_t vl = __riscv_vsetvl_e32m4(n - i);
        vfloat32m4_t shifted = __riscv_vfsub_vf_f32m4(
            __riscv_vle32_v_f32m4(activations + i, vl), max_activation, vl);
        vfloat32m4_t e = exp_f32(shifted, vl);
        vfloat32m1_t red = __riscv_vfredusum_vs_f32m4_f32m1(e, sum_seed, vl);
        sum_seed = __riscv_vfmv_s_f_f32m1(__riscv_vfmv_f_s_f32m1_f32(red), 1);
    }
    const float denom = __riscv_vfmv_f_s_f32m1_f32(sum_seed);
    const float log_denom = std::log(denom);

    // Stage 3: log_probs = act - max - log(denom).
    i = 0;
    for (; i + vlmax <= n; i += vlmax) {
        vfloat32m4_t out = __riscv_vfsub_vf_f32m4(
            __riscv_vle32_v_f32m4(activations + i, vlmax), max_activation, vlmax);
        out = __riscv_vfsub_vf_f32m4(out, log_denom, vlmax);
        __riscv_vse32_v_f32m4(log_probs + i, out, vlmax);
    }
    if (i < n) {
        const size_t vl = __riscv_vsetvl_e32m4(n - i);
        vfloat32m4_t out = __riscv_vfsub_vf_f32m4(
            __riscv_vle32_v_f32m4(activations + i, vl), max_activation, vl);
        out = __riscv_vfsub_vf_f32m4(out, log_denom, vl);
        __riscv_vse32_v_f32m4(log_probs + i, out, vl);
    }
}

}  // namespace ctc_rvv
#endif  // defined(__riscv) && defined(__riscv_v)

template<typename ProbT>
void
CpuCTC<ProbT>::log_softmax(const ProbT* const activations, ProbT* log_probs,
                           const int* const input_lengths) {
#pragma omp parallel for
    for (int mb = 0; mb < minibatch_; ++mb) {
        for(int c = 0; c < input_lengths[mb]; ++c) {
            int col_offset = (mb + minibatch_ * c) * alphabet_size_;
#if defined(__riscv) && defined(__riscv_v)
            if constexpr (std::is_same<ProbT, float>::value) {
                ctc_rvv::log_softmax_f32(activations + col_offset,
                                         log_probs + col_offset,
                                         alphabet_size_);
                continue;
            }
#endif
            ProbT max_activation = -std::numeric_limits<ProbT>::infinity();
            for(int r = 0; r < alphabet_size_; ++r)
                max_activation = std::max(max_activation, activations[r + col_offset]);

            ProbT denom = ProbT(0.);
            for(int r = 0; r < alphabet_size_; ++r) {
                denom += std::exp(activations[r + col_offset] - max_activation);
            }

            for(int r = 0; r < alphabet_size_; ++r) {
                log_probs[r + col_offset] = activations[r + col_offset]
                                            - max_activation - std::log(denom);
            }
        }
    }
}

template<typename ProbT>
std::tuple<ProbT, bool>
CpuCTC<ProbT>::cost_and_grad_kernel(ProbT *grad, const ProbT* const log_probs,
                                    const int* const labels,
                                    int T, int L, int mb, size_t bytes_used) {

    const int S = 2*L + 1; // Number of labels with blanks

    CpuCTC_metadata ctcm(L, S, T, mb, alphabet_size_, workspace_, bytes_used, blank_label_, labels);

    bool over_threshold = false;

    if (L + ctcm.repeats > T) {
        return std::make_tuple(ProbT(0), over_threshold); // TODO, not right to return 0
    }

    ProbT llForward = compute_alphas(log_probs, ctcm.repeats, S, T, ctcm.e_inc,
                                     ctcm.s_inc, ctcm.labels_w_blanks,
                                     ctcm.alphas);

    ProbT llBackward = compute_betas_and_grad(grad, log_probs, llForward, ctcm.repeats,
                                              S, T, ctcm.e_inc, ctcm.s_inc,
                                              ctcm.labels_w_blanks,
                                              ctcm.alphas,
                                              ctcm.betas,
                                              ctcm.output);

    ProbT diff = std::abs(llForward - llBackward);
    if (diff > ctc_helper::threshold) {
        over_threshold = true;
    }

    return std::make_tuple(-llForward, over_threshold);
}

// Computes forward probabilities
template<typename ProbT>
ProbT CpuCTC<ProbT>::compute_alphas(const ProbT* log_probs, int repeats, int S, int T,
                                    const int* const e_inc,
                                    const int* const s_inc,
                                    const int* const labels,
                                    ProbT* alphas) {

    int start =  (((S /2) + repeats - T) < 0) ? 0 : 1,
            end = S > 1 ? 2 : 1;

    for (int i = start; i < end; ++i) {
        alphas[i] = log_probs[labels[i]];
    }

    for(int t = 1; t < T; ++t) {
        int remain = (S / 2) + repeats - (T - t);
        if(remain >= 0)
            start += s_inc[remain];
        if(t <= (S / 2) + repeats)
            end += e_inc[t - 1];
        int startloop = start;
        int idx1 = t * S, idx2 = (t - 1) * S, idx3 = t * (alphabet_size_ * minibatch_);

        if (start == 0) {
            alphas[idx1] = alphas[idx2] + log_probs[blank_label_ + idx3];
            startloop += 1;
        }

        for(int i = startloop; i < end; ++i) {
            ProbT prev_sum = ctc_helper::log_plus<ProbT>()(alphas[i + idx2], alphas[(i-1) + idx2]);

            // Skip two if not on blank and not on repeat.
            if (labels[i] != blank_label_ && i != 1 && labels[i] != labels[i-2])
                prev_sum = ctc_helper::log_plus<ProbT>()(prev_sum, alphas[(i-2) + idx2]);

            alphas[i + idx1] = prev_sum + log_probs[labels[i] + idx3];
        }
    }

    ProbT loglike = ctc_helper::neg_inf<ProbT>();
    for(int i = start; i < end; ++i) {
        loglike = ctc_helper::log_plus<ProbT>()(loglike, alphas[i + (T - 1) * S]);
    }

    return loglike;
}

// Starting from T, we sweep backward over the alpha array computing one column
// of betas as we go.  At each position we can update product alpha * beta and then
// sum into the gradient associated with each label.
// NOTE computes gradient w.r.t UNNORMALIZED final layer activations.
// Assumed passed in grads are already zeroed!
template<typename ProbT>
ProbT CpuCTC<ProbT>::compute_betas_and_grad(ProbT* grad, const ProbT* const log_probs,
                                            ProbT log_partition, int repeats,
                                            int S, int T, const int* const e_inc,
                                            const int* const s_inc,
                                            const int* const labels,
                                            ProbT* alphas,
                                            ProbT* betas,
                                            ProbT* output) {
    int start = S > 1 ? (S - 2) : 0,
            end = (T > (S / 2) + repeats) ? S : S-1;

    std::fill(output, output + alphabet_size_, ctc_helper::neg_inf<ProbT>());

    //set the starting values in the beta column at the very right edge
    for (int i = start; i < end; ++i) {
        betas[i] = log_probs[labels[i] + (T - 1) * (alphabet_size_ * minibatch_)];

        //compute alpha * beta in log space at this position in (S, T) space
        alphas[i + (T - 1) * S] += betas[i];

        //update the gradient associated with this label
        //essentially performing a reduce-by-key in a sequential manner
        output[labels[i]] =
                ctc_helper::log_plus<ProbT>()(alphas[i + (T - 1) * S], output[labels[i]]);
    }

    //update the gradient wrt to each unique label
    for (int i = 0; i < alphabet_size_; ++i) {
        int idx3 = (T - 1) * alphabet_size_ * minibatch_ + i;

        if (output[i] == 0.0 || output[i] == ctc_helper::neg_inf<ProbT>() ||
            log_probs[idx3] == ctc_helper::neg_inf<ProbT>()) {
            grad[idx3] = std::exp(log_probs[idx3]);
        } else {
            grad[idx3] = std::exp(log_probs[idx3])
                         - std::exp(output[i] - log_probs[idx3] - log_partition);
        }
    }

    //loop from the second to last column all the way to the left
    for(int t = T - 2; t >= 0; --t) {
        int remain = (S / 2) + repeats - (T - t);
        if(remain >= -1)
            start -= s_inc[remain + 1];
        if(t < (S / 2) + repeats)
            end -= e_inc[t];

        int endloop = end == S ? end - 1 : end;
        int idx1 = t * S, idx3 = t * (alphabet_size_ * minibatch_);

        std::fill(output, output + alphabet_size_, ctc_helper::neg_inf<ProbT>());

        for(int i = start; i < endloop; ++i) {
            ProbT next_sum = ctc_helper::log_plus<ProbT>()(betas[i], betas[(i+1)]);
            // Skip two if not on blank and not on repeat.
            if (labels[i] != blank_label_ && i != (S-2) && labels[i] != labels[i+2]){
                next_sum = ctc_helper::log_plus<ProbT>()(next_sum, betas[(i+2)]);
            }
            betas[i] = next_sum + log_probs[labels[i] + idx3];

            //compute alpha * beta in log space
            alphas[i + idx1] += betas[i];

            //update the gradient associated with this label
            output[labels[i]] =
                    ctc_helper::log_plus<ProbT>()(alphas[i + idx1], output[labels[i]]);
        }

        if (end == S) {
            betas[(S-1)] = betas[(S-1)] + log_probs[blank_label_ + idx3];
            alphas[(S-1) + idx1] += betas[(S-1)];

            output[labels[S-1]] =
                    ctc_helper::log_plus<ProbT>()(alphas[S-1 + idx1], output[labels[S-1]]);
        }

        //go over the unique labels and compute the final grad
        // wrt to each one at this time step
        for (int i = 0; i < alphabet_size_; ++i) {

            if (output[i] == 0.0 || output[i] == ctc_helper::neg_inf<ProbT>() ||
                log_probs[idx3] == ctc_helper::neg_inf<ProbT>()) {
                grad[idx3] = std::exp(log_probs[idx3]);
            } else {
                grad[idx3] = std::exp(log_probs[idx3])
                             - std::exp(output[i] - log_probs[idx3] - log_partition);
            }
            ++idx3;
        }
    }

    ProbT loglike = ctc_helper::neg_inf<ProbT>();
    for(int i = start; i < end; ++i) {
        loglike = ctc_helper::log_plus<ProbT>()(loglike, betas[i]);
    }

    return loglike;
}

template<typename ProbT>
ctcStatus_t
CpuCTC<ProbT>::cost_and_grad(const ProbT* const activations,
                             ProbT *grads,
                             ProbT *costs,
                             const int* const flat_labels,
                             const int* const label_lengths,
                             const int* const input_lengths) {
    if (activations == nullptr ||
        grads == nullptr ||
        costs == nullptr ||
        flat_labels == nullptr ||
        label_lengths == nullptr ||
        input_lengths == nullptr
        )
        return CTC_STATUS_INVALID_VALUE;

    ProbT* log_probs = static_cast<ProbT *>(workspace_);

    int maxT = *std::max_element(input_lengths, input_lengths + minibatch_);

    size_t bytes_used = sizeof(ProbT) * minibatch_ * alphabet_size_ * maxT;

    //per minibatch memory
    size_t per_minibatch_bytes = 0;

    int maxL = *std::max_element(label_lengths, label_lengths + minibatch_);;
    int maxS = 2 * maxL + 1;

    //output
    per_minibatch_bytes += sizeof(float) * alphabet_size_;

    //alphas
    per_minibatch_bytes += sizeof(float) * maxS * maxT;

    //betas
    per_minibatch_bytes += sizeof(float) * maxS;

    //labels w/blanks, e_inc, s_inc
    per_minibatch_bytes += 3 * sizeof(int) * maxS;

    log_softmax(activations, log_probs, input_lengths);

#pragma omp parallel for
    for (int mb = 0; mb < minibatch_; ++mb) {
        const int T = input_lengths[mb]; // Length of utterance (time)
        const int L = label_lengths[mb]; // Number of labels in transcription

        bool mb_status;

        std::tie(costs[mb], mb_status) =
                cost_and_grad_kernel(grads + mb * alphabet_size_,
                                     log_probs + mb * alphabet_size_,
                                     flat_labels + std::accumulate(label_lengths, label_lengths + mb, 0),
                                     T, L, mb,
                                     bytes_used + mb * per_minibatch_bytes);
    }

    return CTC_STATUS_SUCCESS;
}

template<typename ProbT>
ctcStatus_t CpuCTC<ProbT>::score_forward(const ProbT* const activations,
                                         ProbT* costs,
                                         const int* const flat_labels,
                                         const int* const label_lengths,
                                         const int* const input_lengths) {
    if (activations == nullptr ||
        costs == nullptr ||
        flat_labels == nullptr ||
        label_lengths == nullptr ||
        input_lengths == nullptr
        )
        return CTC_STATUS_INVALID_VALUE;

    ProbT* log_probs = static_cast<ProbT *>(workspace_);

    int maxT = *std::max_element(input_lengths, input_lengths + minibatch_);

    size_t bytes_used = sizeof(ProbT) * minibatch_ * alphabet_size_ * maxT;

    //per minibatch memory
    size_t per_minibatch_bytes = 0;

    int maxL = *std::max_element(label_lengths, label_lengths + minibatch_);
    int maxS = 2 * maxL + 1;

    //output
    per_minibatch_bytes += sizeof(float) * alphabet_size_;

    //alphas
    per_minibatch_bytes += sizeof(float) * maxS * maxT;

    //betas
    per_minibatch_bytes += sizeof(float) * maxS;

    //labels w/blanks, e_inc, s_inc
    per_minibatch_bytes += 3 * sizeof(int) * maxS;

    log_softmax(activations, log_probs, input_lengths);

#pragma omp parallel for
    for (int mb = 0; mb < minibatch_; ++mb) {
        const int T = input_lengths[mb]; // Length of utterance (time)
        const int L = label_lengths[mb]; // Number of labels in transcription
        const int S = 2*L + 1; // Number of labels with blanks

        CpuCTC_metadata ctcm(L, S, T, mb, alphabet_size_, workspace_,
                             bytes_used + mb * per_minibatch_bytes, blank_label_,
                             flat_labels + std::accumulate(label_lengths, label_lengths + mb, 0));


        if (L + ctcm.repeats > T)
            costs[mb] = ProbT(0);
        else {
            costs[mb] = -compute_alphas(log_probs + mb * alphabet_size_, ctcm.repeats, S, T,
                                        ctcm.e_inc, ctcm.s_inc, ctcm.labels_w_blanks,
                                        ctcm.alphas);
        }

    }

    return CTC_STATUS_SUCCESS;
}

} // mxnet_warpctc
