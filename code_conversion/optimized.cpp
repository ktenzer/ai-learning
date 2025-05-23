#include <iostream>
#include <iomanip>
#include <chrono>

double calculate(long long iterations, double param1, double param2) {
    double result = 1.0;
    for (long long i = 1; i <= iterations; i++) {
        double j = i * param1 - param2;
        result -= (1.0 / j);
        j = i * param1 + param2;
        result += (1.0 / j);
    }
    return result;
}

int main() {
    long long iterations = 100'000'000;
    double param1 = 4.0, param2 = 1.0;

    auto start_time = std::chrono::high_resolution_clock::now();
    double result = calculate(iterations, param1, param2) * 4.0;
    auto end_time = std::chrono::high_resolution_clock::now();

    std::cout << std::fixed << std::setprecision(12) << "Result: " << result << std::endl;
    std::cout << std::fixed << std::setprecision(6) << "Execution Time: " << std::chrono::duration<double>(end_time - start_time).count() << " seconds" << std::endl;

    return 0;
}
