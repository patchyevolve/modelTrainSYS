#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <cstring>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

float* get_data(py::array_t<float>& arr) {
    auto buf = arr.request();
    return static_cast<float*>(buf.ptr);
}

const float* get_const_data(const py::array_t<float>& arr) {
    auto buf = arr.request();
    return static_cast<const float*>(buf.ptr);
}

int get_size(const py::array_t<float>& arr) {
    auto buf = arr.request();
    int n = 1;
    for (auto s : buf.shape) n *= s;
    return n;
}

void apply_repetition_penalty(py::array_t<float> logits, const std::vector<int>& ids,
                               float penalty, int window) {
    float* data = get_data(logits);
    int n = get_size(logits);
    if (penalty == 1.0f || ids.empty()) return;

    int start = std::max(0, (int)ids.size() - window);
    std::vector<bool> seen(n, false);
    for (int i = start; i < (int)ids.size(); i++) {
        int tid = ids[i];
        if (tid >= 0 && tid < n && !seen[tid]) {
            seen[tid] = true;
            data[tid] /= penalty;
        }
    }
}

void apply_top_k(py::array_t<float> logits, int k) {
    float* data = get_data(logits);
    int n = get_size(logits);
    if (k <= 0 || k >= n) return;

    std::vector<float> vals(data, data + n);
    std::nth_element(vals.begin(), vals.begin() + k - 1, vals.end(),
                     std::greater<float>());
    float kth = vals[k - 1];
    for (int i = 0; i < n; i++) {
        if (data[i] < kth) data[i] = -std::numeric_limits<float>::infinity();
    }
}

void apply_top_p_inplace(float* data, int n, float p) {
    if (p <= 0.0f || p >= 1.0f) return;

    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return data[a] > data[b]; });

    float max_val = data[idx[0]];
    float sum = 0.0f;
    float inv_max = 1.0f;
    bool overflow = (max_val > 80.0f);
    if (!overflow) inv_max = std::exp(-max_val);

    for (int i = 0; i < n; i++) {
        float prob = overflow ? 1.0f : std::exp(data[idx[i]]) * inv_max;
        sum += prob;
        if (sum - prob > p) {
            for (int j = i; j < n; j++) data[idx[j]] = -std::numeric_limits<float>::infinity();
            break;
        }
    }
}

void apply_top_p(py::array_t<float> logits, float p) {
    apply_top_p_inplace(get_data(logits), get_size(logits), p);
}

void apply_min_p_inplace(float* data, int n, float min_p) {
    if (min_p <= 0.0f) return;

    float max_val = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; i++) {
        if (data[i] > max_val) max_val = data[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += std::exp(data[i] - max_val);

    float threshold = (1.0f / sum) * min_p;
    for (int i = 0; i < n; i++) {
        float prob = std::exp(data[i] - max_val) / sum;
        if (prob < threshold) data[i] = -std::numeric_limits<float>::infinity();
    }
}

void apply_min_p(py::array_t<float> logits, float min_p) {
    apply_min_p_inplace(get_data(logits), get_size(logits), min_p);
}

int sample_from(float* data, int n) {
    float max_val = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; i++) {
        if (data[i] > max_val) max_val = data[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        data[i] = std::exp(data[i] - max_val);
        sum += data[i];
    }

    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, sum);
    float sample = dist(rng);
    float cumulative = 0.0f;
    for (int i = 0; i < n; i++) {
        cumulative += data[i];
        if (sample <= cumulative) return i;
    }
    return n - 1;
}

int sample(py::array_t<float> logits) {
    return sample_from(get_data(logits), get_size(logits));
}

int generate_token(py::array_t<float> logits, const std::vector<int>& ids,
                   float temperature, int top_k, float top_p, float min_p,
                   float repetition_penalty, int rep_range) {
    float* data = get_data(logits);
    int n = get_size(logits);

    if (repetition_penalty != 1.0f && !ids.empty()) {
        int start = std::max(0, (int)ids.size() - rep_range);
        std::vector<bool> seen(n, false);
        for (int i = start; i < (int)ids.size(); i++) {
            int tid = ids[i];
            if (tid >= 0 && tid < n && !seen[tid]) {
                seen[tid] = true;
                data[tid] /= repetition_penalty;
            }
        }
    }

    if (temperature > 0.0f && temperature != 1.0f) {
        float inv_temp = 1.0f / temperature;
        for (int i = 0; i < n; i++) data[i] *= inv_temp;
    }

    apply_top_k(logits, top_k);
    apply_top_p_inplace(data, n, top_p);
    apply_min_p_inplace(data, n, min_p);

    return sample_from(data, n);
}

PYBIND11_MODULE(logit_processors, m) {
    m.doc() = "C++ logit processors for efficient text generation";
    m.def("apply_repetition_penalty", &apply_repetition_penalty, "Apply repetition penalty");
    m.def("apply_top_k", &apply_top_k, "Apply top-k filtering");
    m.def("apply_top_p", &apply_top_p, "Apply top-p (nucleus) filtering");
    m.def("apply_min_p", &apply_min_p, "Apply min-p filtering");
    m.def("sample", &sample, "Sample from logits");
    m.def("generate_token", &generate_token, "Single generation step with all processors");
}
