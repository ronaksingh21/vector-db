// Standalone correctness test for HNSW::l2_distance.
//
// l2_distance was moved from private to public in hnsw.h specifically so this
// test can call it directly, rather than adding a separate test-only wrapper
// method with the same signature (which would buy no real encapsulation --
// it would still have to accept std::vector<int8_t> either way).
//
// This exists to independently verify the scalar (#else) path here on x86,
// and gives the NEON (#ifdef __ARM_NEON) path on the Pi the same reference
// values to check against later.
#include "hnsw.h"
#include <vector>
#include <cstdint>
#include <cmath>
#include <iostream>

int main() {
    // 128 dims, matching real SIFT dimensionality. Values mix positive,
    // negative, zero, and edge values at +/-127 (the clamp range quantize()
    // produces) so diffs cover the full range up to the max possible 254,
    // stressing the NEON widening logic's assumptions (see l2_distance).
    std::vector<int8_t> a = {
        127, -127, 0, 1, -1, 64, -64, 100, -100, 50, -50, 127, -127, 0, 5, -5,
        127, -127, 0, 1, -1, 64, -64, 100, -100, 50, -50, 127, -127, 0, 5, -5,
        127, -127, 0, 1, -1, 64, -64, 100, -100, 50, -50, 127, -127, 0, 5, -5,
        127, -127, 0, 1, -1, 64, -64, 100, -100, 50, -50, 127, -127, 0, 5, -5,
        127, -127, 0, 1, -1, 64, -64, 100, -100, 50, -50, 127, -127, 0, 5, -5,
        127, -127, 0, 1, -1, 64, -64, 100, -100, 50, -50, 127, -127, 0, 5, -5,
        127, -127, 0, 1, -1, 64, -64, 100, -100, 50, -50, 127, -127, 0, 5, -5,
        127, -127, 0, 1, -1, 64, -64, 100, -100, 50, -50, 127, -127, 0, 5, -5,
    };
    std::vector<int8_t> b = {
        -127, 127, 0, -1, 1, -64, 64, -100, 100, -50, 50, 0, 0, 127, -5, 5,
        -127, 127, 0, -1, 1, -64, 64, -100, 100, -50, 50, 0, 0, 127, -5, 5,
        -127, 127, 0, -1, 1, -64, 64, -100, 100, -50, 50, 0, 0, 127, -5, 5,
        -127, 127, 0, -1, 1, -64, 64, -100, 100, -50, 50, 0, 0, 127, -5, 5,
        -127, 127, 0, -1, 1, -64, 64, -100, 100, -50, 50, 0, 0, 127, -5, 5,
        -127, 127, 0, -1, 1, -64, 64, -100, 100, -50, 50, 0, 0, 127, -5, 5,
        -127, 127, 0, -1, 1, -64, 64, -100, 100, -50, 50, 0, 0, 127, -5, 5,
        -127, 127, 0, -1, 1, -64, 64, -100, 100, -50, 50, 0, 0, 127, -5, 5,
    };

    if (a.size() != 128 || b.size() != 128) {
        std::cout << "FAIL: test vectors are not 128-dimensional (a=" << a.size()
                   << ", b=" << b.size() << ")\n";
        return 1;
    }

    // Source of truth: a plain scalar loop, written independently of
    // HNSW::l2_distance -- it must not call the method under test.
    long long expected_sum_sq = 0;
    for (size_t i = 0; i < a.size(); i++) {
        int d = (int)a[i] - (int)b[i];
        expected_sum_sq += (long long)d * d;
    }
    float expected = std::sqrt((float)expected_sum_sq);

    HNSW index(16, 5, 200);
    float actual = index.l2_distance(a, b);

    std::cout << "expected sum_sq: " << expected_sum_sq << "\n";
    std::cout << "expected l2:     " << expected << "\n";
    std::cout << "actual l2:       " << actual << "\n";

    bool pass = std::fabs(expected - actual) < 1e-3f;
    std::cout << (pass ? "PASS" : "FAIL") << "\n";
    return pass ? 0 : 1;
}
