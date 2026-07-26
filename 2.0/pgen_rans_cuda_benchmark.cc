// SPDX-License-Identifier: GPL-3.0-or-later

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "pgen_rans_codec.h"
#include "pgen_rans_container.h"
#include "pgen_rans_cpu.h"
#include "pgen_rans_cuda.h"

namespace {

using pgen_rans::CodecParams;
using pgen_rans::ContainerReader;
using pgen_rans::CpuBlockDecoder;
using pgen_rans::CudaBlockDecoder;
using pgen_rans::CudaDecodeTimings;
using pgen_rans::EncodedBlockView;
using pgen_rans::PackedWordCt;

struct Options {
  std::string input_fname;
  uint32_t block_ct = 80;
  uint32_t batch_block_ct = 8;
  uint32_t iteration_ct = 3;
  uint32_t cpu_thread_ct = 0;
};

void PrintUsage(FILE* out) {
  fputs(
      "Usage:\n"
      "  pgen_rans_cuda_benchmark <rans.pgen> [options]\n"
      "\n"
      "Options:\n"
      "  --blocks <n>         Deterministic block strata (default 80).\n"
      "  --batch-blocks <n>   Blocks decoded per CUDA batch (default 8).\n"
      "  --iterations <n>     CUDA decode passes per batch (default 3).\n"
      "  --cpu-threads <n>    Workers used for exact CPU validation.\n",
      out);
}

bool ParseU32(const char* text, uint32_t min_value, uint32_t max_value,
              uint32_t* value) {
  if ((!text) || (!text[0])) {
    return false;
  }
  errno = 0;
  char* endp = nullptr;
  const unsigned long long parsed = strtoull(text, &endp, 10);
  if (errno || (!endp) || endp[0] || (parsed < min_value) ||
      (parsed > max_value)) {
    return false;
  }
  *value = static_cast<uint32_t>(parsed);
  return true;
}

bool ParseOptions(int argc, char** argv, Options* opts) {
  if ((argc < 2) || (!strcmp(argv[1], "--help"))) {
    PrintUsage((argc < 2) ? stderr : stdout);
    return argc >= 2;
  }
  opts->input_fname = argv[1];
  for (int arg_idx = 2; arg_idx < argc; ++arg_idx) {
    if (!strcmp(argv[arg_idx], "--help")) {
      PrintUsage(stdout);
      exit(0);
    }
    if (arg_idx + 1 == argc) {
      fprintf(stderr, "Error: Missing value after %s.\n", argv[arg_idx]);
      return false;
    }
    const char* option = argv[arg_idx++];
    const char* value = argv[arg_idx];
    if (!strcmp(option, "--blocks")) {
      if (!ParseU32(value, 1, UINT32_MAX, &(opts->block_ct))) {
        fprintf(stderr, "Error: Invalid --blocks value '%s'.\n", value);
        return false;
      }
    } else if (!strcmp(option, "--batch-blocks")) {
      if (!ParseU32(value, 1, 1024, &(opts->batch_block_ct))) {
        fprintf(stderr, "Error: Invalid --batch-blocks value '%s'.\n",
                value);
        return false;
      }
    } else if (!strcmp(option, "--iterations")) {
      if (!ParseU32(value, 1, 1000, &(opts->iteration_ct))) {
        fprintf(stderr, "Error: Invalid --iterations value '%s'.\n", value);
        return false;
      }
    } else if (!strcmp(option, "--cpu-threads")) {
      if (!ParseU32(value, 1, 1024, &(opts->cpu_thread_ct))) {
        fprintf(stderr, "Error: Invalid --cpu-threads value '%s'.\n",
                value);
        return false;
      }
    } else {
      fprintf(stderr, "Error: Unknown option '%s'.\n", option);
      return false;
    }
  }
  return true;
}

std::vector<uint32_t> SelectBlocks(uint32_t block_ct,
                                   uint32_t requested_block_ct) {
  const uint32_t selected_ct = std::min(block_ct, requested_block_ct);
  std::vector<uint32_t> result;
  result.reserve(selected_ct);
  for (uint32_t stratum_idx = 0; stratum_idx != selected_ct;
       ++stratum_idx) {
    const uint64_t numerator =
        static_cast<uint64_t>(2 * stratum_idx + 1) * block_ct;
    result.push_back(
        static_cast<uint32_t>(numerator / (2 * selected_ct)));
  }
  return result;
}

bool CheckCuda(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) {
    return true;
  }
  fprintf(stderr, "Error: %s: %s\n", operation,
          cudaGetErrorString(status));
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  Options opts;
  if (!ParseOptions(argc, argv, &opts)) {
    return 2;
  }
  if (!strcmp(argv[1], "--help")) {
    return 0;
  }
  ContainerReader reader;
  std::string error;
  if (!reader.Open(opts.input_fname, &error)) {
    fprintf(stderr, "Error: %s\n", error.c_str());
    return 1;
  }
  const auto& params = reader.params();
  if (params.state_ct != 32) {
    fputs("Error: CUDA prototype requires 32 rANS states.\n", stderr);
    return 1;
  }
  CudaBlockDecoder cuda_decoder;
  if (!cuda_decoder.available()) {
    fputs("Error: No CUDA device is available.\n", stderr);
    return 1;
  }
  CpuBlockDecoder cpu_decoder(opts.cpu_thread_ct);
  if (!cpu_decoder.thread_ct()) {
    fputs("Error: Could not start CPU validation workers.\n", stderr);
    return 1;
  }
  const std::vector<uint32_t> selected_blocks =
      SelectBlocks(params.block_ct, opts.block_ct);
  const uint32_t word_stride = PackedWordCt(params.sample_ct);
  const size_t maximum_batch_variant_ct =
      static_cast<size_t>(opts.batch_block_ct) * params.block_variant_ct;
  if (maximum_batch_variant_ct >
      std::numeric_limits<size_t>::max() / word_stride) {
    fputs("Error: CUDA batch output size exceeds platform limits.\n",
          stderr);
    return 1;
  }
  const size_t maximum_output_word_ct =
      maximum_batch_variant_ct * word_stride;
  uint64_t* device_output = nullptr;
  if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_output),
                            maximum_output_word_ct * sizeof(uint64_t)),
                 "cudaMalloc(output)")) {
    return 1;
  }

  const CodecParams codec_params = {
      params.state_ct, params.scale_bits};
  uint64_t block_bytes = 0;
  uint64_t variant_ct = 0;
  uint64_t calls_per_pass = 0;
  double block_read_seconds = 0.0;
  double decoder_wall_seconds = 0.0;
  double upload_seconds = 0.0;
  double anchor_seconds = 0.0;
  double target_seconds = 0.0;
  uint64_t checksum = 0;
  int return_code = 1;

  for (size_t batch_start = 0; batch_start < selected_blocks.size();
       batch_start += opts.batch_block_ct) {
    const size_t batch_size =
        std::min(static_cast<size_t>(opts.batch_block_ct),
                 selected_blocks.size() - batch_start);
    std::vector<std::vector<uint8_t>> block_storage(batch_size);
    std::vector<EncodedBlockView> block_views(batch_size);
    std::vector<const EncodedBlockView*> block_ptrs(batch_size);
    uint32_t batch_variant_ct = 0;
    for (size_t batch_offset = 0; batch_offset != batch_size;
         ++batch_offset) {
      const uint32_t block_idx =
          selected_blocks[batch_start + batch_offset];
      const auto read_start = std::chrono::steady_clock::now();
      if (!reader.ReadBlockView(
              block_idx, &(block_storage[batch_offset]),
              &(block_views[batch_offset]), &error)) {
        fprintf(stderr, "Error: %s\n", error.c_str());
        goto cleanup;
      }
      block_read_seconds +=
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - read_start)
              .count();
      block_ptrs[batch_offset] = &(block_views[batch_offset]);
      batch_variant_ct += block_views[batch_offset].variant_ct();
      block_bytes += reader.block_index()[block_idx].byte_ct;
    }
    const size_t batch_output_word_ct =
        static_cast<size_t>(batch_variant_ct) * word_stride;
    std::vector<uint64_t> host_output(batch_output_word_ct);
    for (uint32_t iteration_idx = 0;
         iteration_idx != opts.iteration_ct; ++iteration_idx) {
      CudaDecodeTimings timings;
      const auto decode_start = std::chrono::steady_clock::now();
      if (!cuda_decoder.Decode(
              block_ptrs, params.sample_ct, codec_params, device_output,
              maximum_output_word_ct, nullptr, &timings, &error)) {
        fprintf(stderr, "Error: %s\n", error.c_str());
        goto cleanup;
      }
      decoder_wall_seconds +=
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - decode_start)
              .count();
      upload_seconds += timings.upload_milliseconds / 1000.0;
      anchor_seconds += timings.anchor_milliseconds / 1000.0;
      target_seconds += timings.target_milliseconds / 1000.0;
      calls_per_pass +=
          iteration_idx ? 0 : timings.genotype_call_ct;
    }
    if (!CheckCuda(cudaMemcpy(
                       host_output.data(), device_output,
                       batch_output_word_ct * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost),
                   "cudaMemcpy(output)")) {
      goto cleanup;
    }
    size_t batch_output_variant = 0;
    for (size_t batch_offset = 0; batch_offset != batch_size;
         ++batch_offset) {
      const EncodedBlockView& block = block_views[batch_offset];
      std::vector<uint64_t> expected(
          static_cast<size_t>(block.variant_ct()) * word_stride);
      if (!cpu_decoder.Decode(
              block, params.sample_ct, codec_params, expected.data(),
              expected.size(), &error)) {
        fprintf(stderr, "Error: %s\n", error.c_str());
        goto cleanup;
      }
      const uint64_t* observed =
          host_output.data() + batch_output_variant * word_stride;
      if (memcmp(expected.data(), observed,
                 expected.size() * sizeof(uint64_t))) {
        fprintf(stderr,
                "Error: CUDA/CPU mismatch in block starting at variant "
                "%u.\n",
                block.first_variant());
        goto cleanup;
      }
      checksum ^=
          observed[(block.first_variant() + batch_output_variant) %
                   expected.size()];
      batch_output_variant += block.variant_ct();
      variant_ct += block.variant_ct();
    }
    fprintf(stderr, "\rValidated %zu/%zu sampled blocks.",
            std::min(batch_start + batch_size, selected_blocks.size()),
            selected_blocks.size());
    fflush(stderr);
  }
  {
    const long double decoded_calls =
        static_cast<long double>(calls_per_pass) * opts.iteration_ct;
    const double kernel_seconds = anchor_seconds + target_seconds;
    printf("\nConditional-rANS CUDA benchmark\n");
    printf("  samples:                    %u\n", params.sample_ct);
    printf("  variants:                   %llu\n",
           static_cast<unsigned long long>(variant_ct));
    printf("  blocks:                     %zu / %u\n",
           selected_blocks.size(), params.block_ct);
    printf("  blocks per CUDA batch:      %u\n", opts.batch_block_ct);
    printf("  decode iterations:          %u\n", opts.iteration_ct);
    printf("  checksum:                   %016llx\n",
           static_cast<unsigned long long>(checksum));
    printf("\nTiming\n");
    printf("  rANS-PGEN block bytes:      %llu\n",
           static_cast<unsigned long long>(block_bytes));
    printf("  rANS-PGEN read seconds:     %.6f\n", block_read_seconds);
    printf("  H2D upload seconds:         %.6f\n", upload_seconds);
    printf("  anchor kernel seconds:      %.6f\n", anchor_seconds);
    printf("  target kernel seconds:      %.6f\n", target_seconds);
    printf("  decoder wall seconds:       %.6f\n", decoder_wall_seconds);
    printf("  kernel billion calls/sec:   %.3Lf\n",
           (kernel_seconds > 0.0)
               ? decoded_calls / kernel_seconds / 1.0e9L
               : 0.0L);
    printf("  decoder billion calls/sec:  %.3Lf\n",
           (decoder_wall_seconds > 0.0)
               ? decoded_calls / decoder_wall_seconds / 1.0e9L
               : 0.0L);
  }
  return_code = 0;

cleanup:
  cudaFree(device_output);
  return return_code;
}
