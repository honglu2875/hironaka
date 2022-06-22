#include "../hironaka/cpp/cppUtil.h"
#include <iostream>

int main() {
    int points[] = {    \
        7, 5, 3, 8,     \
        8, 1, 8, 18,    \
        8, 3, 17, 8,    \
        11, 11, 1, 19,  \
        11, 12, 18, 6,  \
        16, 11, 5, 26,  \
        -1, -1, -1, -1,
        1,2,3,4};
    int len = sizeof(points)/sizeof(*points)/4;
    int newPoints[len * 4];

    std::fill_n(newPoints, len*4, -1);
    _getNewtonPolytope_approx_brute_force(points, 1, 4, len, newPoints);

    for (int i=0;i<len*4;i++) {
        std::cout << newPoints[i] << " ";
    }

    return 0;
}
