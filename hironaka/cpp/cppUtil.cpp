#include <iostream>
#define locate(b,i,j) b*m*n+i*n+j

void _getNewtonPolytope_approx_brute_force(long* points, int batchNum, int m, int n, long* newPoints) {
    /*
    *   Get the maximum subset where every point (x_1,...,x_n) is not "larger" than any point (y_1,...,y_n) from the original set (satisfying x_i<=y_i for all i).
    *   This is a brute force implementation, O(m^2*n) where m is the number of points and n is the number of coordinates.
    *   This can be used as
    *       - either a substitute of Newton polytope algorithm (more points -> slightly more wasteful on further operations, but the result has one single point if and only if the Newton polytope has one single vertex),
    *       - or a part of Newton polytope algorithm (first get the convex hull, second take the inner part by invoking this function).
    *
    *   points: The array of points. 2d but flattened to 1d in the contiguous fashion.
    *   batchNum: batch size.
    *   n: dimension.
    *   m: number of points in one batch.
    *   (in python it is a 3d numpy array. (batch, m, n))
    *   newPoints: The array that stores
    *
    *   ASSUMPTIONS:
    *   number of points in each batch is no more than m.
    *   in each batch, the first appearance of "-1" represents the end of the point list.
    */

    bool isContained = true;
    //for (int i=0;i<24;i++) {
    //    std::cout<<points[i]<<std::endl;
    //}
    for (int b=0;b<batchNum;b++){
        int counter = 0;

        for (int i=0;i<m;i++) {
            if (points[locate(b,i,0)]==-1) break; // End of data points in batch b
            for (int j=0;j<m;j++) {
                if (points[locate(b,j,0)]==-1) break; // End of data points in batch b
                if (j==i) continue;

                isContained = true;
                for (int k=0;k<n;k++) {
                    if (points[locate(b,i,k)]<points[locate(b,j,k)]) {
                        isContained = false;
                        break;
                    }
                }
                if (isContained) break;
            }
            if (!isContained) {
                for (int k=0;k<n;k++) {
                    newPoints[locate(b,counter,k)] = points[locate(b,i,k)];
                }
                //std::cout<<i<<std::endl;
                counter++;
            }
        }



    }
}

extern "C" {
    void getNewtonPolytope_approx(long* points, int batchNum, int m, int n, long* newPoints) {
        _getNewtonPolytope_approx_brute_force(points, batchNum, m, n, newPoints); // TODO: is there a more optimal way?
    }

}
