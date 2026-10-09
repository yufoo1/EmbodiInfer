// Optional cuBLASLt bridge. No Torch ABI, model metadata or scheduler dependencies.
#include <cublasLt.h>
#include <cstdint>

struct ReferencePlan {
  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t weight = nullptr, input = nullptr, output = nullptr;
  cublasLtMatmulAlgo_t algorithm{};
  size_t workspace = 1048576;
  int status = 0;
};

#define PLAN_CHECK(expression)                     \
  do {                                            \
    plan->status = static_cast<int>(expression);   \
    if (plan->status != 0) return plan;            \
  } while (false)

extern "C" void* reference_plan_create(int width, int outputs, int reference_rows,
                                      int rows, void* bias, int algorithm_id,
                                      int tile, int stages) {
  auto* plan = new ReferencePlan;
  PLAN_CHECK(cublasLtCreate(&plan->handle));
  PLAN_CHECK(cublasLtMatmulDescCreate(&plan->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t transpose = CUBLAS_OP_T;
  PLAN_CHECK(cublasLtMatmulDescSetAttribute(plan->operation,
      CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)));
  if (bias) {
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    PLAN_CHECK(cublasLtMatmulDescSetAttribute(plan->operation,
        CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    PLAN_CHECK(cublasLtMatmulDescSetAttribute(plan->operation,
        CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
  }
  PLAN_CHECK(cublasLtMatrixLayoutCreate(&plan->weight, CUDA_R_16BF, width, outputs, width));
  PLAN_CHECK(cublasLtMatrixLayoutCreate(&plan->input, CUDA_R_16BF, width, reference_rows, width));
  PLAN_CHECK(cublasLtMatrixLayoutCreate(&plan->output, CUDA_R_16BF, outputs, reference_rows, outputs));
  cublasLtMatmulPreference_t preference = nullptr;
  PLAN_CHECK(cublasLtMatmulPreferenceCreate(&preference));
  plan->status = cublasLtMatmulPreferenceSetAttribute(preference,
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &plan->workspace, sizeof(plan->workspace));
  cublasLtMatmulHeuristicResult_t reference{};
  int count = 0;
  if (plan->status == 0) {
    plan->status = cublasLtMatmulAlgoGetHeuristic(plan->handle, plan->operation,
        plan->weight, plan->input, plan->output, plan->output,
        preference, 1, &reference, &count);
  }
  if (plan->status != 0 || count == 0) {
    cublasLtMatmulPreferenceDestroy(preference);
    if (plan->status == 0) plan->status = CUBLAS_STATUS_NOT_SUPPORTED;
    return plan;
  }
  plan->algorithm = reference.algo;
  cublasLtMatrixLayoutDestroy(plan->input);
  cublasLtMatrixLayoutDestroy(plan->output);
  plan->input = nullptr;
  plan->output = nullptr;
  plan->status = cublasLtMatrixLayoutCreate(&plan->input, CUDA_R_16BF, width, rows, width);
  if (plan->status == 0) {
    plan->status = cublasLtMatrixLayoutCreate(&plan->output, CUDA_R_16BF, outputs, rows, outputs);
  }
  if (plan->status != 0) {
    cublasLtMatmulPreferenceDestroy(preference);
    return plan;
  }

  int split = 1, reduction = 0;
  size_t written = 0;
  auto status = cublasLtMatmulAlgoConfigGetAttribute(&reference.algo,
      CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &split, sizeof(split), &written);
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulAlgoConfigGetAttribute(&reference.algo,
        CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reduction, sizeof(reduction), &written);
  }
  if (status != CUBLAS_STATUS_SUCCESS) {
    cublasLtMatmulPreferenceDestroy(preference);
    plan->status = status;
    return plan;
  }

  cublasLtMatmulAlgo_t proposed{};
  bool explicit_algorithm = algorithm_id >= 0;
  if (explicit_algorithm) {
    plan->workspace = 33554432;
    status = cublasLtMatmulAlgoInit(plan->handle, CUBLAS_COMPUTE_32F, CUDA_R_32F,
        CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, algorithm_id, &proposed);
    if (status == CUBLAS_STATUS_SUCCESS) {
      status = cublasLtMatmulAlgoConfigSetAttribute(&proposed,
          CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
      status = cublasLtMatmulAlgoConfigSetAttribute(&proposed,
          CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof(stages));
    }
  } else {
    cublasLtMatmulHeuristicResult_t candidate{};
    count = 0;
    status = cublasLtMatmulAlgoGetHeuristic(plan->handle, plan->operation,
        plan->weight, plan->input, plan->output, plan->output,
        preference, 1, &candidate, &count);
    proposed = candidate.algo;
    if (status == CUBLAS_STATUS_SUCCESS && count == 0) status = CUBLAS_STATUS_NOT_SUPPORTED;
  }
  cublasLtMatmulPreferenceDestroy(preference);
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulAlgoConfigSetAttribute(&proposed,
        CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &split, sizeof(split));
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulAlgoConfigSetAttribute(&proposed,
        CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reduction, sizeof(reduction));
  }
  cublasLtMatmulHeuristicResult_t check{};
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulAlgoCheck(plan->handle, plan->operation,
        plan->weight, plan->input, plan->output, plan->output, &proposed, &check);
  }
  if (status == CUBLAS_STATUS_SUCCESS) status = check.state;
  if (status == CUBLAS_STATUS_SUCCESS && check.workspaceSize > plan->workspace) {
    status = CUBLAS_STATUS_NOT_SUPPORTED;
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    plan->algorithm = proposed;
  } else if (explicit_algorithm) {
    plan->status = status;
  }
  // Without explicit tuning, retain the serial algorithm when retuning fails.
  // Execution can still report NOT_SUPPORTED, which requests true serial calls.
  return plan;
}

extern "C" int reference_plan_execute(void* pointer, void* input, void* weight,
                                     void* output, void* workspace, void* stream) {
  auto* plan = static_cast<ReferencePlan*>(pointer);
  if (plan->status != 0) return plan->status;
  float alpha = 1, beta = 0;
  return cublasLtMatmul(plan->handle, plan->operation, &alpha, weight, plan->weight,
      input, plan->input, &beta, output, plan->output, output, plan->output,
      &plan->algorithm, workspace, plan->workspace, static_cast<cudaStream_t>(stream));
}

extern "C" void reference_plan_destroy(void* pointer) {
  auto* plan = static_cast<ReferencePlan*>(pointer);
  if (!plan) return;
  if (plan->weight) cublasLtMatrixLayoutDestroy(plan->weight);
  if (plan->input) cublasLtMatrixLayoutDestroy(plan->input);
  if (plan->output) cublasLtMatrixLayoutDestroy(plan->output);
  if (plan->operation) cublasLtMatmulDescDestroy(plan->operation);
  if (plan->handle) cublasLtDestroy(plan->handle);
  delete plan;
}
