#include "triton/Analysis/Alias.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {

AliasInfo AliasInfo::join(const AliasInfo &lhs, const AliasInfo &rhs) {
  if (lhs == rhs)
    return lhs;
  AliasInfo ret;
  for (auto value : lhs.allocs) {
    ret.insert(value);
  }
  for (auto value : rhs.allocs) {
    ret.insert(value);
  }
  return ret;
}

#if NOT_COMPATIBLE_WITH_LATEST_LLLVM
ChangeResult SharedMemoryAliasAnalysis::visitOperation(
    Operation *op, ArrayRef<LatticeElement<AliasInfo> *> operands) {
  AliasInfo aliasInfo;
  bool pessimistic = true;
  if (maybeSharedAllocationOp(op)) {
    // These ops may allocate a new shared memory buffer.
    auto result = op->getResult(0);
    if (isSharedEncoding(result)) {
      // FIXME(Keren): extract and insert are always alias for now
      if (auto extractSliceOp = dyn_cast<triton::gpu::ExtractSliceOp>(op)) {
        // extract_slice %src, %index
        aliasInfo = AliasInfo(operands[0]->getValue());
      } else if (auto insertSliceOp =
                     dyn_cast<triton::gpu::InsertSliceAsyncOp>(op)) {
        // insert_slice_async %src, %dst, %index
        aliasInfo = AliasInfo(operands[1]->getValue());
      } else {
        aliasInfo.insert(result);
      }
      pessimistic = false;
    }
  }

  if (pessimistic) {
    return markAllPessimisticFixpoint(op->getResults());
  }
  // Join all latice elements
  ChangeResult result = ChangeResult::NoChange;
  for (Value value : op->getResults()) {
    result |= getLatticeElement(value).join(aliasInfo);
  }
  return result;
}

AliasResult SharedMemoryAliasAnalysis::alias(Value lhs, Value rhs) {
  // TODO: implement
  return AliasResult::MayAlias;
}

ModRefResult SharedMemoryAliasAnalysis::getModRef(Operation *op,
                                                  Value location) {
  // TODO: implement
  return ModRefResult::getModAndRef();
}
#endif

} // namespace mlir
