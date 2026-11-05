#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace strata::core {
inline int emitted_prefix_rows(int accepted_rows, int64_t remaining, const int32_t* out,
                               const std::vector<int64_t>& eos) {
    const int rows = (int) std::min<int64_t>(accepted_rows, std::max<int64_t>(remaining, 0));
    for (int i = 0; i < rows; ++i)
        if (std::find(eos.begin(), eos.end(), (int64_t) out[i]) != eos.end()) return i + 1;
    return rows;
}

inline int accepted_commit_last(int accepted_last, int64_t remaining, const int32_t* out,
                                const std::vector<int64_t>& eos, bool merged, const char* scope) {
    static const bool fix = [] { const char* e = std::getenv("STRATA_COMMIT_EMITTED_PREFIX"); return e && e[0] == '1'; }();
    static const bool trace = [] { const char* e = std::getenv("STRATA_COMMIT_TRACE"); return e && e[0] == '1'; }();
    if (merged || (!fix && !trace)) return accepted_last;
    const int rows = emitted_prefix_rows(accepted_last + 1, remaining, out, eos);
    if (trace && rows != accepted_last + 1)
        std::fprintf(stderr, "strata commit boundary: %s accepted=%d emitted=%d remaining=%lld fix=%d\n",
                     scope, accepted_last + 1, rows, (long long) remaining, fix ? 1 : 0);
    // Only emitted rows become state; the last output stays the next pending input.
    return fix ? rows - 1 : accepted_last;
}
}
