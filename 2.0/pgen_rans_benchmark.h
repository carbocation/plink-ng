// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef PGEN_RANS_BENCHMARK_H_
#define PGEN_RANS_BENCHMARK_H_

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace pgen_rans {

struct TimingSummary {
  double median = 0.0;
  double minimum = 0.0;
  double maximum = 0.0;
  double median_absolute_deviation = 0.0;
};

inline double Median(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const size_t midpoint = values.size() / 2;
  if (values.size() & 1) {
    return values[midpoint];
  }
  return (values[midpoint - 1] + values[midpoint]) / 2.0;
}

inline TimingSummary SummarizeTimings(const std::vector<double>& samples) {
  TimingSummary result;
  if (samples.empty()) {
    return result;
  }
  result.median = Median(samples);
  const auto bounds = std::minmax_element(samples.begin(), samples.end());
  result.minimum = *bounds.first;
  result.maximum = *bounds.second;
  std::vector<double> deviations;
  deviations.reserve(samples.size());
  for (const double sample : samples) {
    deviations.push_back(std::fabs(sample - result.median));
  }
  result.median_absolute_deviation = Median(std::move(deviations));
  return result;
}

}  // namespace pgen_rans

#endif  // PGEN_RANS_BENCHMARK_H_
