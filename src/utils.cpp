#include <iostream>
#include <cmath>
#include <array>

//using Matrix3x3 = std::array<std::array<double, 3>, 3>;
using namespace std;

// Returns the 3x3 rotation matrix (HA, delta in radians)
void EarthRotationEffect(double HA, double delta, double R[3][3]) {
    R[0][0] =  sin(HA);
    R[0][1] =  cos(HA);
    R[0][2] =  0.0;

    R[1][0] = -sin(delta) * cos(HA);
    R[1][1] =  sin(delta) * sin(HA);
    R[1][2] =  cos(delta);

    R[2][0] =  cos(delta) * cos(HA);
    R[2][1] = -cos(delta) * sin(HA);
    R[2][2] =  sin(delta);
}
/*
int main() {
    double HA = 1.0;     // Hour angle in radians
    double delta = 0.5;  // Declination in radians

    double R[3][3];
    EarthRotationEffect(HA, delta, R);

    // Print the matrix
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            cout << R[i][j] << " ";
        }
        cout << "\n";
    }

    return 0;
}
*/

