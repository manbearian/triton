#include "triton/Analysis/Allocation.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "triton/Analysis/Alias.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <limits>

namespace mlir {

//===----------------------------------------------------------------------===//
// Shared Memory Allocation Analysis
//===----------------------------------------------------------------------===//
namespace triton {
class AllocationAnalysis {
public:
  AllocationAnalysis(Operation *operation, Allocation *allocation)
      : operation(operation), allocation(allocation) {
    run();
  }

private:
  using BufferT = Allocation::BufferT;

  /// Value -> Liveness Range
  /// Use MapVector to ensure determinism.
  using BufferRangeMapT = llvm::MapVector<BufferT *, Interval<size_t>>;
  /// Nodes -> Nodes
  using GraphT = DenseMap<BufferT *, DenseSet<BufferT *>>;

  void run() {
    getValuesAndSizes();
    resolveLiveness();
    computeOffsets();
  }

  /// Initializes explicitly defined shared memory values for a given operation.
  void getExplicitValueSize(Operation *op) {
    // Values returned from scf.yield will not be allocated even though they
    // have the shared encoding.
    // For example: %a = scf.if -> yield
    // %a must be allocated elsewhere by other operations.
    // FIXME(Keren): extract and insert are always alias for now
    if (!maybeSharedAllocationOp(op) || isa<triton::gpu::ExtractSliceOp>(op) ||
        isa<triton::gpu::InsertSliceAsyncOp>(op)) {
      return;
    }

    for (Value result : op->getResults()) {
      if (isSharedEncoding(result)) {
        // Bytes could be a different value once we support padding or other
        // allocation policies.
        auto tensorType = result.getType().dyn_cast<RankedTensorType>();
        auto bytes = tensorType.getNumElements() *
                     tensorType.getElementTypeBitWidth() / 8;
        allocation->addBuffer<BufferT::BufferKind::Explicit>(result, bytes);
      }
    }
  }

  /// Initializes temporary shared memory for a given operation.
  void getScratchValueSize(Operation *op) {
    // TODO(Keren): Add atomic ops
    // TODO(Keren): Add convert ops
    if (auto reduceOp = dyn_cast<triton::ReduceOp>(op)) {
      // TODO(Keren): Reduce with index is not supported yet.
      auto value = op->getOperand(0);
      if (auto tensorType = value.getType().dyn_cast<RankedTensorType>()) {
        auto bytes = tensorType.getNumElements() *
                     tensorType.getElementTypeBitWidth() / 8;
        allocation->addBuffer<BufferT::BufferKind::Scratch>(op, bytes);
      }
    }
  }

  void getValueAlias(Value value, SharedMemoryAliasAnalysis &analysis) {
    if (auto latticeElement = analysis.lookupLatticeElement(value)) {
      auto &info = latticeElement->getValue();
      if (!info.getAllocs().empty()) {
        for (auto alloc : info.getAllocs()) {
          allocation->addAlias(value, alloc);
        }
      }
    }
  }

  /// Extract all shared memory values and their sizes
  void getValuesAndSizes() {
    // Get the alloc values
    operation->walk<WalkOrder::PreOrder>([&](Operation *op) {
      getExplicitValueSize(op);
      getScratchValueSize(op);
    });
    // Get the alias values
    SharedMemoryAliasAnalysis aliasAnalysis(operation->getContext());
    aliasAnalysis.run(operation);
    operation->walk<WalkOrder::PreOrder>([&](Operation *op) {
      for (auto operand : op->getOperands()) {
        getValueAlias(operand, aliasAnalysis);
      }
      for (auto value : op->getResults()) {
        getValueAlias(value, aliasAnalysis);
      }
    });
  }

  /// Computes the liveness range of the allocated value.
  /// Each buffer is allocated only once.
  void resolveExplicitBufferLiveness(
      function_ref<Interval<size_t>(Value value)> getLiveness) {
    for (auto valueBufferIter : allocation->valueBuffer) {
      auto value = valueBufferIter.first;
      auto *buffer = valueBufferIter.second;
      bufferRange[buffer] = getLiveness(value);
    }
  }

  /// Extends the liveness range by unioning the liveness range of the aliased
  /// values because each allocated buffer could be an alias of others, if block
  /// arguments are involved.
  void resolveAliasBufferLiveness(
      function_ref<Interval<size_t>(Value value)> getLiveness) {
    for (auto aliasBufferIter : allocation->aliasBuffer) {
      auto value = aliasBufferIter.first;
      auto buffers = aliasBufferIter.second;
      auto range = getLiveness(value);
      for (auto *buffer : buffers) {
        auto minId = range.start();
        auto maxId = range.end();
        if (bufferRange.count(buffer)) {
          // Extend the allocated buffer's range
          minId = std::min(minId, bufferRange[buffer].start());
          maxId = std::max(maxId, bufferRange[buffer].end());
        }
        bufferRange[buffer] = Interval(minId, maxId);
      }
    }
  }

  /// Computes the liveness range of scratched buffers.
  /// Some operations may have a temporary buffer that is not explicitly
  /// allocated, but is used to store intermediate results.
  void resolveScratchBufferLiveness(
      const DenseMap<Operation *, size_t> &operationId) {
    // Analyze liveness of scratch buffers
    for (auto opScratchIter : allocation->opScratch) {
      // Any scratch memory's live range is the current operation's live
      // range.
      auto *op = opScratchIter.first;
      auto *buffer = opScratchIter.second;
      bufferRange.insert({buffer, Interval(operationId.lookup(op),
                                           operationId.lookup(op) + 1)});
    }
  }

  /// Resolves liveness of all values involved under the root operation.
  void resolveLiveness() {
    // In the SCF dialect, we always have a sequentially nested structure of
    // blocks
    DenseMap<Operation *, size_t> operationId;
    operation->walk<WalkOrder::PreOrder>(
        [&](Operation *op) { operationId[op] = operationId.size(); });

    // Analyze liveness of explicit buffers
    Liveness liveness(operation);
    auto getValueLivenessRange = [&](Value value) {
      auto liveOperations = liveness.resolveLiveness(value);
      auto minId = std::numeric_limits<size_t>::max();
      auto maxId = std::numeric_limits<size_t>::min();
      std::for_each(liveOperations.begin(), liveOperations.end(),
                    [&](Operation *liveOp) {
                      if (operationId[liveOp] < minId) {
                        minId = operationId[liveOp];
                      }
                      if ((operationId[liveOp] + 1) > maxId) {
                        maxId = operationId[liveOp] + 1;
                      }
                    });
      return Interval(minId, maxId);
    };

    resolveExplicitBufferLiveness(getValueLivenessRange);
    resolveAliasBufferLiveness(getValueLivenessRange);
    resolveScratchBufferLiveness(operationId);
  }

  /// Computes the shared memory offsets for all related values.
  /// Paper: Algorithms for Compile-Time Memory Optimization
  /// (https://www.cs.utexas.edu/users/harrison/papers/compile-time.pdf)
  void computeOffsets() {
    SmallVector<BufferT *> buffers;
    for (auto bufferIter : bufferRange) {
      buffers.emplace_back(bufferIter.first);
    }

    DenseMap<BufferT *, size_t> bufferStart;
    calculateStarts(buffers, bufferStart);

    GraphT interference;
    buildInterferenceGraph(buffers, bufferStart, interference);

    allocate(buffers, bufferStart, interference);
  }

  /// Computes the initial shared memory offsets.
  void calculateStarts(const SmallVector<BufferT *> &buffers,
                       DenseMap<BufferT *, size_t> &bufferStart) {
    //  v = values in shared memory
    //  t = triplet of (size, start, end)
    //  shared memory space
    //  -
    //  |         *******t4
    //  | /|\ v2 inserts t4, t5, and t6
    //  |  |
    //  | ******t5         ************t6
    //  | ^^^^^v2^^^^^^
    //  |  |      *********************t2
    //  | \|/ v2 erases t1
    //  | ******t1 ^^^^^^^^^v1^^^^^^^^^ ************t3
    //  |---------------------------------------------| liveness range
    //    1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 ...
    /// Start -> Liveness Range
    using TripleMapT = std::multimap<size_t, Interval<size_t>>;
    TripleMapT tripleMap;
    tripleMap.insert(std::make_pair(0, Interval<size_t>()));
    SmallVector<BufferT *> xBuffers = buffers;
    while (!xBuffers.empty()) {
      auto tripleIt = tripleMap.begin();
      auto size = tripleIt->first;
      auto range = tripleIt->second;
      tripleMap.erase(tripleIt);
      auto bufferIt =
          std::find_if(xBuffers.begin(), xBuffers.end(), [&](auto *buffer) {
            auto xRange = bufferRange[buffer];
            bool res = xRange.intersects(range);
            for (auto val : tripleMap)
              res = res && !val.second.intersects(xRange);
            return res;
          });
      if (bufferIt != xBuffers.end()) {
        auto buffer = *bufferIt;
        auto xSize = buffer->size;
        auto xRange = bufferRange.lookup(buffer);
        bufferStart[buffer] = size;
        tripleMap.insert(
            {size + xSize, Interval{std::max(range.start(), xRange.start()),
                                    std::min(range.end(), xRange.end())}});
        if (range.start() < xRange.start())
          tripleMap.insert({size, Interval{range.start(), xRange.end()}});
        if (xRange.end() < range.end())
          tripleMap.insert({size, Interval{xRange.start(), range.end()}});
        xBuffers.erase(bufferIt);
      }
    }
  }

  /// Builds a graph of all shared memory values. Edges are created between
  /// shared memory values that are overlapping.
  void buildInterferenceGraph(const SmallVector<BufferT *> &buffers,
                              const DenseMap<BufferT *, size_t> &bufferStart,
                              GraphT &interference) {
    for (auto x : buffers) {
      for (auto y : buffers) {
        if (x == y)
          continue;
        auto xStart = bufferStart.lookup(x);
        auto yStart = bufferStart.lookup(y);
        auto xSize = x->size;
        auto ySize = y->size;
        Interval xSizeRange = {xStart, xStart + xSize};
        Interval ySizeRange = {yStart, yStart + ySize};
        auto xOpRange = bufferRange.lookup(x);
        auto yOpRange = bufferRange.lookup(y);
        if (xOpRange.intersects(yOpRange) &&
            xSizeRange.intersects(ySizeRange)) {
          interference[x].insert(y);
        }
      }
    }
  }

  /// Finalizes shared memory offsets considering interference.
  void allocate(const SmallVector<BufferT *> &buffers,
                const DenseMap<BufferT *, size_t> &bufferStart,
                const GraphT &interference) {
    // First-fit graph coloring
    // Neighbors are nodes that interfere with each other.
    // We color a node by finding the index of the first available
    // non-neighboring node or the first neighboring node without any color.
    // Nodes with the same color do not interfere with each other.
    DenseMap<BufferT *, int> colors;
    for (auto value : buffers) {
      colors[value] = (value == buffers[0]) ? 0 : -1;
    }
    SmallVector<bool> available(buffers.size());
    for (auto x : buffers) {
      std::fill(available.begin(), available.end(), true);
      for (auto y : interference.lookup(x)) {
        int color = colors[y];
        if (color >= 0) {
          available[color] = false;
        }
      }
      auto it = std::find(available.begin(), available.end(), true);
      colors[x] = std::distance(available.begin(), it);
    }
    // Finalize allocation
    // color0: [0, 7), [0, 8), [0, 15) -> [0, 7), [0, 8), [0, 15)
    // color1: [7, 9) -> [0 + 1 * 15, 9 + 1 * 15) -> [15, 24)
    // color2: [8, 12) -> [8 + 2 * 15, 12 + 2 * 15) -> [38, 42)
    // TODO(Keren): We are wasting memory here.
    // Nodes with color2 can actually start with 24.
    for (auto x : buffers) {
      size_t adj = 0;
      for (auto y : interference.lookup(x)) {
        adj = std::max(adj, bufferStart.lookup(y) + y->size);
      }
      x->offset = bufferStart.lookup(x) + colors.lookup(x) * adj;
      allocation->sharedMemorySize =
          std::max(allocation->sharedMemorySize, x->offset + x->size);
    }
  }

private:
  Operation *operation;
  Allocation *allocation;
  BufferRangeMapT bufferRange;
};
} // namespace triton

void Allocation::run() { triton::AllocationAnalysis(getOperation(), this); }

} // namespace mlir
