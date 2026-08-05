// SPDX-License-Identifier: LGPL-3.0-or-later

#include "include/pgenlib_read.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "include/plink2_bits.h"

namespace {

using namespace plink2;

[[noreturn]] void Fail(const char* message) {
  fprintf(stderr, "FAIL: %s\n", message);
  exit(1);
}

void Expect(bool condition, const char* message) {
  if (!condition) {
    Fail(message);
  }
}

struct FakeBackendContext {
  bool dense_called = false;
  bool sparse_called = false;
  bool return_dense_from_sparse = false;
};

PglErr GetDense(void* context, const uintptr_t*, const uint32_t*,
                uint32_t sample_ct, uint32_t, uintptr_t* genovec) {
  FakeBackendContext* fake = static_cast<FakeBackendContext*>(context);
  fake->dense_called = true;
  memset(genovec, 0, NypCtToWordCt(sample_ct) * sizeof(uintptr_t));
  AssignNyparrEntry(1, 1, genovec);
  return kPglRetSuccess;
}

PglErr GetSparse(void* context, const uintptr_t*, const uint32_t*,
                 uint32_t, uint32_t, uint32_t max_difflist_len,
                 uintptr_t* genovec, uint32_t* common_genotype,
                 uintptr_t* raregenos, uint32_t* sample_ids,
                 uint32_t* difflist_len) {
  FakeBackendContext* fake = static_cast<FakeBackendContext*>(context);
  fake->sparse_called = true;
  if (fake->return_dense_from_sparse) {
    *common_genotype = UINT32_MAX;
    *difflist_len = 0;
    memset(genovec, 0, sizeof(uintptr_t));
    AssignNyparrEntry(4, 2, genovec);
    return kPglRetSuccess;
  }
  if (max_difflist_len < 3) {
    return kPglRetImproperFunctionCall;
  }
  *common_genotype = 0;
  *difflist_len = 3;
  sample_ids[0] = 2;
  sample_ids[1] = 5;
  sample_ids[2] = 9;
  AssignNyparrEntry(0, 1, raregenos);
  AssignNyparrEntry(1, 2, raregenos);
  AssignNyparrEntry(2, 3, raregenos);
  return kPglRetSuccess;
}

PgenReader MakeReader(PgrHardcallBackend* backend) {
  PgenReader result{};
  PreinitPgr(&result);
  PgenReaderMain* pgrp = &GET_PRIVATE(result, m);
  pgrp->fi.raw_sample_ct = 16;
  pgrp->fi.raw_variant_ct = 1;
  pgrp->hardcall_backend = backend;
  return result;
}

}  // namespace

int main() {
  using namespace plink2;
  FakeBackendContext context;
  PgrHardcallBackend backend{};
  backend.context = &context;
  backend.get = &GetDense;
  backend.get_difflist_or_genovec = &GetSparse;
  PgenReader reader = MakeReader(&backend);
  PgrSampleSubsetIndex pssi;
  PgrClearSampleSubsetIndex(&reader, &pssi);

  uintptr_t genovec[1] = {};
  uintptr_t dosage_present[1] = {};
  uint16_t dosage_main[16] = {};
  uint32_t sample_ids[16] = {};
  uint32_t dosage_ct = UINT32_MAX;
  uint16_t common_dosage = 1;
  Expect(PgrGetDMaybeSparse(
             nullptr, pssi, 16, 0, 8, &reader, genovec, dosage_present,
             dosage_main, &dosage_ct, &common_dosage, sample_ids) ==
             kPglRetSuccess,
         "PgrGetDMaybeSparse rejected the sparse backend");
  Expect(context.sparse_called && !context.dense_called,
         "PgrGetDMaybeSparse did not dispatch to the sparse backend");
  Expect((common_dosage == 0) && (dosage_ct == 3) &&
             (sample_ids[0] == 2) && (sample_ids[1] == 5) &&
             (sample_ids[2] == 9) && (dosage_main[0] == 16384) &&
             (dosage_main[1] == 32768) &&
             (dosage_main[2] == UINT16_MAX),
         "PgrGetDMaybeSparse converted sparse hardcalls incorrectly");

  uintptr_t raregenos[1] = {};
  uint32_t common_genotype = UINT32_MAX;
  uint32_t difflist_len = 0;
  context.sparse_called = false;
  Expect(PgrGetDifflistOrGenovec(
             nullptr, pssi, 16, 8, 0, &reader, genovec,
             &common_genotype, raregenos, sample_ids, &difflist_len) ==
             kPglRetSuccess,
         "PgrGetDifflistOrGenovec rejected the sparse backend");
  Expect(context.sparse_called && (common_genotype == 0) &&
             (difflist_len == 3) &&
             (GetNyparrEntry(raregenos, 0) == 1) &&
             (GetNyparrEntry(raregenos, 1) == 2) &&
             (GetNyparrEntry(raregenos, 2) == 3),
         "PgrGetDifflistOrGenovec sparse backend result mismatch");

  context.return_dense_from_sparse = true;
  context.sparse_called = false;
  common_dosage = 0;
  dosage_ct = UINT32_MAX;
  Expect(PgrGetDMaybeSparse(
             nullptr, pssi, 16, 0, 8, &reader, genovec, dosage_present,
             dosage_main, &dosage_ct, &common_dosage, sample_ids) ==
             kPglRetSuccess,
         "PgrGetDMaybeSparse rejected callback dense fallback");
  Expect(context.sparse_called && (common_dosage == 1) && (!dosage_ct) &&
             (GetNyparrEntry(genovec, 4) == 2),
         "PgrGetDMaybeSparse callback dense fallback mismatch");
  context.return_dense_from_sparse = false;

  backend.get_difflist_or_genovec = nullptr;
  reader = MakeReader(&backend);
  PgrClearSampleSubsetIndex(&reader, &pssi);
  context.dense_called = false;
  context.sparse_called = false;
  common_dosage = 0;
  dosage_ct = UINT32_MAX;
  Expect(PgrGetDMaybeSparse(
             nullptr, pssi, 16, 0, 8, &reader, genovec, dosage_present,
             dosage_main, &dosage_ct, &common_dosage, sample_ids) ==
             kPglRetSuccess,
         "PgrGetDMaybeSparse dense compatibility fallback failed");
  Expect(context.dense_called && !context.sparse_called &&
             (common_dosage == 1) && (!dosage_ct) &&
             (GetNyparrEntry(genovec, 1) == 1),
         "PgrGetDMaybeSparse dense compatibility result mismatch");

  puts("pgenlib_sparse_backend_test: PASS");
  return 0;
}
