
#include <cfloat>
#include <cmath>
#include <climits>
#include <iostream>

// BECAREFUL! do not use -ffast-math on g++
static inline void kahan_step(const double& in, double *sum, double *c) {
    double y = in - c[0];
    double t = sum[0] + y;
    c[0] = (t - sum[0]) - y;
    sum[0] = t;
    return;
}

template<typename T>
size_t fp_diff(const T *in0, const T *in1, const size_t &length, const float& eps_abs, const float& eps_rel, const float& eps_cos) {
    size_t min_abs_idx = (size_t)UINT64_MAX;
    size_t max_abs_idx = (size_t)UINT64_MAX;
    float min_abs_err = FLT_MAX;
    float max_abs_err = -FLT_MAX;
    float min_abs_val0;
    float max_abs_val0;
    float min_abs_val1;
    float max_abs_val1;

    size_t min_rel_idx = (size_t)UINT64_MAX;
    size_t max_rel_idx = (size_t)UINT64_MAX;
    float min_rel_err = FLT_MAX;
    float max_rel_err = -FLT_MAX;
    float min_rel_val0;
    float max_rel_val0;
    float min_rel_val1;
    float max_rel_val1;

    size_t abs_en1_cnt = 0;
    size_t abs_en2_cnt = 0;
    size_t abs_en3_cnt = 0;
    size_t abs_en4_cnt = 0;
    size_t abs_en5_cnt = 0;
    size_t abs_en6_cnt = 0;

    size_t rel_en1_cnt = 0;
    size_t rel_en2_cnt = 0;
    size_t rel_en3_cnt = 0;
    size_t rel_en4_cnt = 0;
    size_t rel_en5_cnt = 0;
    size_t rel_en6_cnt = 0;

    size_t err_val_cnt = 0;

    double sum_ab = 0.;
    double sum_a2 = 0.;
    double sum_b2 = 0.;
    double N = length;
    double rN = 1.0 / N;

    double sum_ab_c = 0.;
    double sum_a2_c = 0.;
    double sum_b2_c = 0.;

    int nan_c = 0;
    int inf_c = 0;

    for (size_t i = 0; i < length; ++i) {
        float val0 = in0[i];
        float val1 = in1[i];

        int nan0 = std::isnan(val0);
        int nan1 = std::isnan(val1);
        int inf0 = std::isinf(val0);
        int inf1 = std::isinf(val1);

        if (nan0 && nan1) {
            nan_c++;
            continue;
        }

        if (inf0 && inf1) {
            inf_c++;
            continue;
        }

        kahan_step((double)val0 * val1 * rN, &sum_ab, &sum_ab_c);
        kahan_step((double)val0 * val0 * rN, &sum_a2, &sum_a2_c);
        kahan_step((double)val1 * val1 * rN, &sum_b2, &sum_b2_c);
        // sum_ab += val0 * val1 * rN;
        // sum_a2 += val0 * val0 * rN;
        // sum_b2 += val1 * val1 * rN;

        float err = val0 - val1;
        float abs_err = fabs(err);
        float rel_err = fabs(val0) > fabs(val1) ? fabs(err / val0) : fabs(err / val1);
        if (abs_err > 1e-1f) {
            ++abs_en1_cnt;
        } else if (abs_err > 1e-2f) {
            ++abs_en2_cnt;
        } else if (abs_err > 1e-3f) {
            ++abs_en3_cnt;
        } else if (abs_err > 1e-4f) {
            ++abs_en4_cnt;
        } else if (abs_err > 1e-5f) {
            ++abs_en5_cnt;
        } else if (abs_err > 1e-6f) {
            ++abs_en6_cnt;
        }
        if (rel_err > 1e-1f) {
            ++rel_en1_cnt;
        } else if (rel_err > 1e-2f) {
            ++rel_en2_cnt;
        } else if (rel_err > 1e-3f) {
            ++rel_en3_cnt;
        } else if (rel_err > 1e-4f) {
            ++rel_en4_cnt;
        } else if (rel_err > 1e-5f) {
            ++rel_en5_cnt;
        } else if (rel_err > 1e-6f) {
            ++rel_en6_cnt;
        }
        if ((abs_err >= eps_abs && rel_err >= eps_rel) || nan0 || nan1 || inf0 || inf1) {
            ++err_val_cnt;
        }
        if (abs_err > max_abs_err) {
            max_abs_idx = i;
            max_abs_err = abs_err;
            max_abs_val0 = val0;
            max_abs_val1 = val1;
        }
        if (rel_err > max_rel_err) {
            max_rel_idx = i;
            max_rel_err = rel_err;
            max_rel_val0 = val0;
            max_rel_val1 = val1;
        }
        if (abs_err < min_abs_err) {
            min_abs_idx = i;
            min_abs_err = abs_err;
            min_abs_val0 = val0;
            min_abs_val1 = val1;
        }
        if (rel_err < min_rel_err) {
            min_rel_idx = i;
            min_rel_err = rel_err;
            min_rel_val0 = val0;
            min_rel_val1 = val1;
        }
    }

    double cos_distance;
    if (nan_c || inf_c) {
        std::cerr << "ERROR nan !!!.";
        cos_distance = -1.;
        return 999;
    } else {
        cos_distance = 1.0 - (sum_ab / (sqrt(sum_a2) * sqrt(sum_b2)));
    }

    std::cerr << "Number of values: " << length << std::endl;
    std::cerr << "Number of equal values: " << length - err_val_cnt << std::endl;
    std::cerr << "Number of not equal values: " << err_val_cnt << std::endl;
    std::cerr << "COS simulate: " << 1-cos_distance << ", with SUM(ab)/N: " << sum_ab << ", SUM(a^2)/N:" << sum_a2 << ", SUM(b^2)/N: " << sum_b2 << std::endl;

    if (min_abs_idx != (size_t)UINT64_MAX) std::cerr << "MIN abs error: " << min_abs_err << " --> " << min_abs_val0 << " vs. " << min_abs_val1 << ", at " << min_abs_idx << std::endl;
    if (max_abs_idx != (size_t)UINT64_MAX) std::cerr << "MAX abs error: " << max_abs_err << " --> " << max_abs_val0 << " vs. " << max_abs_val1 << ", at " << max_abs_idx << std::endl;
    if (min_rel_idx != (size_t)UINT64_MAX) std::cerr << "MIN rel error: " << min_rel_err << " --> " << min_rel_val0 << " vs. " << min_rel_val1 << ", at " << min_rel_idx << std::endl;
    if (max_rel_idx != (size_t)UINT64_MAX) std::cerr << "MAX rel error: " << max_rel_err << " --> " << max_rel_val0 << " vs. " << max_rel_val1 << ", at " << max_rel_idx << std::endl;

    std::cerr << "Statistics of abs error" << std::endl;
    std::cerr << "(..., 1e-1f): " << abs_en1_cnt << std::endl;
    std::cerr << "(..., 1e-2f): " << abs_en2_cnt << std::endl;
    std::cerr << "(..., 1e-3f): " << abs_en3_cnt << std::endl;
    std::cerr << "(..., 1e-4f): " << abs_en4_cnt << std::endl;
    std::cerr << "(..., 1e-5f): " << abs_en5_cnt << std::endl;
    std::cerr << "(..., 1e-6f): " << abs_en6_cnt << std::endl;

    std::cerr << "Statistics of rel error" << std::endl;
    std::cerr << "(..., 1e-1f): " << rel_en1_cnt << std::endl;
    std::cerr << "(..., 1e-2f): " << rel_en2_cnt << std::endl;
    std::cerr << "(..., 1e-3f): " << rel_en3_cnt << std::endl;
    std::cerr << "(..., 1e-4f): " << rel_en4_cnt << std::endl;
    std::cerr << "(..., 1e-5f): " << rel_en5_cnt << std::endl;
    std::cerr << "(..., 1e-6f): " << rel_en6_cnt << std::endl;

    if (std::isnan(cos_distance) || std::isinf(cos_distance)) {
        return err_val_cnt;
    }

    if (cos_distance < eps_cos) {
        return 0;
    } else {
        return err_val_cnt;
    }
}