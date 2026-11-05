#include "accepted_commit.hpp"
#include <cassert>
#include <iostream>

int main() {
    int cases = 0, old_nonprefix = 0;
    for (int accepted = 1; accepted <= 8; ++accepted)
        for (int budget = 1; budget <= 10; ++budget)
            for (int stop = -1; stop < accepted; ++stop) {
                int32_t out[8], inputs[8];
                for (int i = 0; i < 8; ++i) out[i] = 100 + i;
                if (stop >= 0) out[stop] = 999;
                inputs[0] = 77;
                for (int i = 1; i < 8; ++i) inputs[i] = out[i - 1];
                std::vector<int32_t> visible;
                for (int i = 0; i < accepted && (int) visible.size() < budget; ++i) {
                    visible.push_back(out[i]);
                    if (out[i] == 999) break;
                }
                const int keep = strata::core::emitted_prefix_rows(accepted, budget, out, {999});
                assert(keep == (int) visible.size());
                std::vector<int32_t> next{77};next.insert(next.end(), visible.begin(), visible.end());next.push_back(12345);
                assert(std::equal(inputs, inputs + keep, next.begin()));
                if ((size_t) accepted > next.size() || !std::equal(inputs, inputs + accepted, next.begin())) ++old_nonprefix;
                ++cases;
            }
    int32_t out[1]{4};
    assert(strata::core::emitted_prefix_rows(1,0,out,{})==0);
    assert(old_nonprefix>0);
    std::cout << cases << " budget/EOS/prefix cases passed; old invalid prefixes " << old_nonprefix << "\n";
}
