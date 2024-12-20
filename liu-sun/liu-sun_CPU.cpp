#include <algorithm>
#include <cassert>
#include <chrono>
#include <cinttypes>
#include <deque>
#include <iostream>
#include <limits>
#include <list>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>

int num_threads;

namespace std {

// We need to make our own hash specialization since std::pair does not have one
template <>
struct hash<std::pair<uint16_t, uint16_t>> {
  size_t operator()(const std::pair<uint16_t, uint16_t> &key) const noexcept {
    // Pack the two 16-bit ints into a single 32-bit int and then has that
    static std::hash<uint32_t> hasher;
    uint32_t tohash = static_cast<uint32_t>(key.first);
    tohash |= (static_cast<uint32_t>(key.second) << 16);
    return hasher(tohash);
  }
};

}  // namespace std

using NodeIdx = uint32_t;
using ArcIdx = uint32_t;
using BlockIdx = uint16_t;
using Time = uint32_t;
using Dist = int16_t;
using Cap = int64_t;
using Flow = int64_t;

// What's this
using Term = int64_t;

enum TermType : int32_t { SOURCE = 0, SINK = 1 };

enum NodeArcSorting : int32_t {
  LIFO = 0,
  FIFO = 1,
  CAP_ASC = 2,
  CAP_DESC = 3,
  HEAD_ASC = 4,
  NEAREST = 5,
  CLUSTER = 6
};
class ParallelGraph {
  // Forward decls.
  struct Node;
  struct Arc;
  struct BoundarySegment;
  struct GraphBlock;

 public:
  static const NodeIdx INVALID_NODE =
      ~NodeIdx(0);  // -1 for signed type, max. value for unsigned type
  static const ArcIdx INVALID_ARC =
      ~ArcIdx(0);  // -1 for signed type, max. value for unsigned type
  static const ArcIdx TERMINAL_ARC = INVALID_ARC - 1;
  static const ArcIdx ORPHAN_ARC = INVALID_ARC - 2;
  static const Cap INACTIVE_ARC = -1;

  ParallelGraph(size_t expected_nodes, size_t expected_arcs,
                size_t expected_blocks);

  NodeIdx add_node(size_t num = 1, BlockIdx block = 0);

  void add_tweights(NodeIdx i, Term cap_source, Term cap_sink);

  void add_edge(NodeIdx i, NodeIdx j, Cap cap, Cap rev_cap);

  TermType what_segment(NodeIdx i, TermType default_segment = SOURCE) const;

  Flow maxflow();
  void init_maxflow();

  inline size_t get_node_num() const noexcept { return nodes.size(); }
  inline size_t get_arc_num() const noexcept { return arcs.size(); }

  NodeArcSorting node_arc_sorting;

  //   std::chrono::duration<double> ph1_dur;
  //   std::chrono::duration<double> bs_dur;
  //   std::chrono::duration<double> ph2_dur;

 private:
  std::vector<Node> nodes;
  std::vector<Arc> arcs;

  std::vector<BlockIdx> node_blocks;

  std::unordered_map<std::pair<BlockIdx, BlockIdx>,
                     std::vector<std::pair<ArcIdx, Cap>>>
      boundary_arcs;
  std::list<BoundarySegment> boundary_segments;
  std::vector<BlockIdx> block_idxs;

  std::vector<GraphBlock> blocks;

  struct BoundarySegment {
    const std::vector<std::pair<ArcIdx, Cap>> &arcs;
    BlockIdx i;
    BlockIdx j;
    int32_t potential_activations;
  };

  struct GraphBlock {
    std::vector<Node> &nodes;
    std::vector<Arc> &arcs;
    std::vector<BlockIdx> &node_blocks;

    BlockIdx self;
    bool locked;
    bool initialized;

    Flow flow;

    NodeIdx first_active, last_active;
    std::deque<NodeIdx> orphan_nodes;

    Time time;

    GraphBlock(std::vector<Node> &nodes, std::vector<Arc> &arcs,
               std::vector<BlockIdx> &node_blocks, BlockIdx self)
        : nodes(nodes),
          arcs(arcs),
          node_blocks(node_blocks),
          self(self),
          locked(false),
          initialized(false),
          flow(0),
          first_active(INVALID_NODE),
          last_active(INVALID_NODE),
          orphan_nodes(),
          time(0) {}

    Flow maxflow();
    void init_maxflow();

    void make_active(NodeIdx i);
    void make_front_orphan(NodeIdx i);
    void make_back_orphan(NodeIdx i);

    NodeIdx next_active();

    void augment(ArcIdx middle);
    Term tree_bottleneck(NodeIdx start, bool source_tree) const;
    void augment_tree(NodeIdx start, Term bottleneck, bool source_tree);

    ArcIdx grow_search_tree(NodeIdx start);
    template <bool source>
    ArcIdx grow_search_tree_impl(NodeIdx start);

    void process_orphan(NodeIdx i);
    template <bool souce>
    void process_orphan_impl(NodeIdx i);

    inline ArcIdx sister_idx(ArcIdx a) const noexcept { return a ^ 1; }
    inline Arc &sister(ArcIdx a) { return arcs[sister_idx(a)]; }
    inline const Arc &sister(ArcIdx a) const { return arcs[sister_idx(a)]; }

    inline ArcIdx sister_or_arc_idx(ArcIdx a, bool sis) const noexcept {
      return a ^ (ArcIdx)sis;
    }
    inline Arc &sister_or_arc(ArcIdx a, bool sis) {
      return arcs[sister_or_arc_idx(a, sis)];
    }
    inline const Arc &sister_or_arc(ArcIdx a, bool sis) const {
      return arcs[sister_or_arc_idx(a, sis)];
    }

    inline Node &head_node(const Arc &a) { return nodes[a.head]; }
    inline Node &head_node(ArcIdx a) { return head_node(arcs[a]); }
    inline const Node &head_node(const Arc &a) const { return nodes[a.head]; }
    inline const Node &head_node(ArcIdx a) const { return head_node(arcs[a]); }
  };

  // #pragma pack(1)
  struct Node {
    ArcIdx first;         // First out-going arc.
    ArcIdx parent;        // Arc to parent node in search tree
    NodeIdx next_active;  // Index of next active node (or itself if this is the
                          // last one)

    Time timestamp;  // Timestamp showing when dist was computed.
    Dist dist;       // Distance to terminal.

    Term tr_cap;  // If tr_cap > 0 then tr_cap is residual capacity of the arc
                  // SOURCE->node. Otherwise         -tr_cap is residual
                  // capacity of the arc node->SINK.

    bool is_sink : 1;  // flag showing if the node is in the source or sink tree
                       // (if parent!=NULL)

    Node()
        : first(INVALID_ARC),
          parent(INVALID_ARC),
          next_active(INVALID_NODE),
          timestamp(0),
          dist(0),
          tr_cap(0),
          is_sink(false) {}
  };

  struct Arc {
    NodeIdx head;  // Node this arc points to.
    ArcIdx next;   // Next arc with the same originating node

    Cap r_cap;  // Residual capacity

    Arc() : head(INVALID_NODE), next(INVALID_ARC), r_cap(0) {}

    Arc(NodeIdx _head, ArcIdx _next, Cap _r_cap)
        : head(_head), next(_next), r_cap(_r_cap) {}
  };
  // #pragma pack()

  ArcIdx add_half_edge(NodeIdx from, NodeIdx to, Cap cap);

  inline std::pair<BlockIdx, BlockIdx> block_key(BlockIdx i,
                                                 BlockIdx j) const noexcept;

  std::pair<std::list<BoundarySegment>, BlockIdx> next_boundary_segment_set();

  inline ArcIdx sister_idx(ArcIdx a) const noexcept { return a ^ 1; }
  inline Arc &sister(ArcIdx a) { return arcs[sister_idx(a)]; }
  inline const Arc &sister(ArcIdx a) const { return arcs[sister_idx(a)]; }

  bool should_activate(NodeIdx i, NodeIdx j);

  inline Node &head_node(const Arc &a) { return nodes[a.head]; }
  inline Node &head_node(ArcIdx a) { return head_node(arcs[a]); }
  inline const Node &head_node(const Arc &a) const { return nodes[a.head]; }
  inline const Node &head_node(ArcIdx a) const { return head_node(arcs[a]); }
};

ParallelGraph::ParallelGraph(size_t expected_nodes, size_t expected_arcs,
                             size_t expected_blocks)
    : nodes(),
      arcs(),
      node_blocks(),
      boundary_arcs(),
      boundary_segments(),
      blocks(),
      node_arc_sorting(LIFO) {
  nodes.reserve(expected_nodes);
  arcs.reserve(2 * expected_arcs);
  node_blocks.reserve(expected_nodes);
  blocks.reserve(expected_blocks);
}

inline NodeIdx ParallelGraph::add_node(size_t num, BlockIdx block) {
  assert(nodes.size() == node_blocks.size());
  NodeIdx crnt = nodes.size();

  nodes.resize(crnt + num);
  node_blocks.resize(crnt + num, block);
  if (block >= blocks.size()) {
    // We assume that we always have blocks 0,1,2,...,N
    for (int b = blocks.size(); b <= block; ++b) {
      blocks.emplace_back(nodes, arcs, node_blocks, b);
      block_idxs.push_back(b);
    }
  }
  return crnt;
}

inline void ParallelGraph::add_tweights(NodeIdx i, Term cap_source,
                                        Term cap_sink) {
  assert(i >= 0 && i < nodes.size());
  assert(node_blocks[i] < blocks.size());
  Term delta = nodes[i].tr_cap;
  if (delta > 0) {
    cap_source += delta;
  } else {
    cap_sink -= delta;
  }
  blocks[node_blocks[i]].flow += std::min(cap_source, cap_sink);
  nodes[i].tr_cap = cap_source - cap_sink;
}

inline void ParallelGraph::add_edge(NodeIdx i, NodeIdx j, Cap cap,
                                    Cap rev_cap) {
  assert(i >= 0 && i < nodes.size());
  assert(j >= 0 && j < nodes.size());
  assert(i != j);
  assert(cap >= 0);
  assert(rev_cap >= 0);

  BlockIdx bi = node_blocks[i];
  BlockIdx bj = node_blocks[j];
  if (bi == bj) {
    add_half_edge(i, j, cap);
    add_half_edge(j, i, rev_cap);
  } else {
    const ArcIdx a1 = add_half_edge(i, j, INACTIVE_ARC);
    const ArcIdx a2 = add_half_edge(j, i, INACTIVE_ARC);
    const auto key = block_key(bi, bj);
    boundary_arcs[key].push_back(std::make_pair(a1, cap));
    boundary_arcs[key].push_back(std::make_pair(a2, rev_cap));
  }
}

inline ArcIdx ParallelGraph::add_half_edge(NodeIdx from, NodeIdx to, Cap cap) {
  ArcIdx ai;

  // Create new arc.
  ai = arcs.size();
  arcs.emplace_back(to, nodes[from].first, cap);
  nodes[from].first = ai;
  return ai;
}

inline TermType ParallelGraph::what_segment(NodeIdx i,
                                            TermType default_segment) const {
  assert(i >= 0 && i < nodes.size());
  if (nodes[i].parent != INVALID_ARC) {
    return (nodes[i].is_sink) ? SINK : SOURCE;
  } else {
    return default_segment;
  }
}

inline Flow ParallelGraph::maxflow() {
  Flow flow = 0;

  std::atomic<BlockIdx> processed_blocks = 0;
  std::vector<std::thread> threads;

  init_maxflow();

  // Solve all base blocks.
  for (unsigned int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&]() {
      const BlockIdx num_blocks = blocks.size();
      BlockIdx crnt = processed_blocks.fetch_add(1);
      while (crnt < num_blocks) {
        blocks[crnt].maxflow();
        crnt = processed_blocks.fetch_add(1);
      }
    });
  }
  for (auto &th : threads) {
    if (th.joinable()) {
      th.join();
    }
  }

  // pha1 end
  // bs starts

  // Build list of boundary segments
  for (const auto &ba : boundary_arcs) {
    BlockIdx i, j;
    std::tie(i, j) = ba.first;
    boundary_segments.push_back({ba.second, i, j, 0});
  }

  // Count number of potential activations for each boundary segment and sort
  for (auto &bs : boundary_segments) {
    int32_t potential_activations = 0;
    for (const auto &a : bs.arcs) {
      Arc &arc = arcs[a.first];
      Arc &sister_arc = sister(a.first);
      if (should_activate(arc.head, sister_arc.head)) {
        potential_activations++;
      }
    }
    bs.potential_activations = potential_activations;
  }

  boundary_segments.sort([](const auto &bs1, const auto &bs2) {
    return bs1.potential_activations > bs2.potential_activations;
  });

  // bs ends
  // ph2 starts

  threads.clear();
  std::mutex lock;
  for (unsigned int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&]() {
      BlockIdx crnt;
      std::list<BoundarySegment> boundary_set;

      while (true) {
        lock.lock();

        std::tie(boundary_set, crnt) = next_boundary_segment_set();
        if (boundary_set.empty()) {
          lock.unlock();
          break;
        }

        auto &block = blocks[crnt];
        block.locked = true;

        lock.unlock();

        // Activate boundary arcs
        for (const auto &bs : boundary_set) {
          for (const auto &a : bs.arcs) {
            Arc &arc = arcs[a.first];
            Arc &sister_arc = sister(a.first);

            arc.r_cap = a.second;

            if (should_activate(arc.head, sister_arc.head)) {
              block.make_active(arc.head);
            }
          }
        }

        block.maxflow();

        lock.lock();
        block.locked = false;
        lock.unlock();
      }
    });
  }
  for (auto &th : threads) {
    if (th.joinable()) {
      th.join();
    }
  }

  // ph2 ends

  // Sum up all subgraph flows
  std::sort(block_idxs.begin(), block_idxs.end());
  auto last = std::unique(block_idxs.begin(), block_idxs.end());
  for (auto iter = block_idxs.begin(); iter != last; ++iter) {
    flow += blocks[*iter].flow;
  }
  return flow;
}

inline std::pair<BlockIdx, BlockIdx> ParallelGraph::block_key(
    BlockIdx i, BlockIdx j) const noexcept {
  return i < j ? std::make_pair(i, j) : std::make_pair(j, i);
}

inline std::pair<std::list<ParallelGraph::BoundarySegment>, BlockIdx>
ParallelGraph::next_boundary_segment_set() {
  // NOTE: We assume the global lock is grabbed at this point so no other
  // threads are scanning
  std::list<BoundarySegment> out;
  BlockIdx out_idx = 0;

  // Scan until we find a boundary segment with unlocked blocks
  auto iter = boundary_segments.begin();
  for (; iter != boundary_segments.end(); ++iter) {
    const auto &bs = *iter;
    if (!blocks[block_idxs[bs.i]].locked && !blocks[block_idxs[bs.j]].locked) {
      // Found boundary between two unlocked blocks
      out_idx = block_idxs[bs.i];
      BlockIdx j = block_idxs[bs.j];
      // Unite blocks
      std::replace(block_idxs.begin(), block_idxs.end(), j, out_idx);
      blocks[out_idx].time = std::max(blocks[out_idx].time, blocks[j].time);
      blocks[out_idx].flow += blocks[j].flow;
      break;
    }
  }
  // If we are not at the end, move all relevant boundary segments to output
  // Note that iter starts at the found boundary segment or at the end
  while (iter != boundary_segments.end()) {
    const auto &bs = *iter;
    if (block_idxs[bs.i] == out_idx && block_idxs[bs.j] == out_idx) {
      out.push_back(bs);
      auto to_erase = iter;
      ++iter;
      boundary_segments.erase(to_erase);
    } else {
      ++iter;
    }
  }
  return std::make_pair(out, out_idx);
}

inline Flow ParallelGraph::GraphBlock::maxflow() {
  NodeIdx crnt_node = INVALID_NODE;

  // main loop
  while (true) {
    NodeIdx i = crnt_node;

    // Check if we are already exploring a valid active node
    if (i != INVALID_NODE) {
      Node &n = nodes[i];
      n.next_active = INVALID_NODE;
      if (n.parent == INVALID_ARC) {
        // Active node was not valid so don't explore after all
        i = INVALID_NODE;
      }
    }

    // If we are not already exploring a node try to get a new one
    if (i == INVALID_NODE) {
      i = next_active();
      if (i == INVALID_NODE) {
        // No more nodes to explore so we are done
        break;
      }
    }

    // At this point i must point to a valid active node
    ArcIdx source_sink_connector = grow_search_tree(i);

    time++;

    if (source_sink_connector != INVALID_ARC) {
      // Growth was aborted because we found a node from the other tree
      nodes[i].next_active = i;  // Mark as active
      crnt_node = i;

      augment(source_sink_connector);

      std::deque<NodeIdx> crnt_orphan_nodes =
          std::move(orphan_nodes);  // Snapshot of current ophans
      for (NodeIdx orphan : crnt_orphan_nodes) {
        process_orphan(orphan);
        // If any additional orphans were added during processing, we process
        // them immediately This leads to a significant decrease of the overall
        // runtime
        while (!orphan_nodes.empty()) {
          NodeIdx o = orphan_nodes.front();
          orphan_nodes.pop_front();
          process_orphan(o);
        }
      }
    } else {
      crnt_node = INVALID_NODE;
    }
  }

  return flow;
}

inline void ParallelGraph::init_maxflow() {
  assert(nodes.size() == node_blocks.size());

  for (auto &b : blocks) {
    if (!b.initialized) {
      b.first_active = INVALID_NODE;
      b.last_active = INVALID_NODE;
      b.orphan_nodes.clear();
      b.time = 0;
    }
  }

  for (size_t i = 0; i < nodes.size(); ++i) {
    auto block_idx = node_blocks[i];
    auto &block = blocks[block_idx];

    if (block.initialized) {
      continue;
    }

    Node &n = nodes[i];
    n.next_active = INVALID_NODE;
    n.timestamp = 0;
    if (n.tr_cap != 0) {
      // Node n is connected to the source or sink
      n.is_sink = n.tr_cap < 0;  // Negative capacity goes to sink
      n.parent = TERMINAL_ARC;
      n.dist = 1;

      block.make_active(i);
    } else {
      n.parent = INVALID_ARC;
    }
  }

  for (auto &b : blocks) {
    b.initialized = true;
  }
}

inline void ParallelGraph::GraphBlock::init_maxflow() {
  assert(nodes.size() == node_blocks.size());

  if (initialized) {
    return;
  }

  first_active = INVALID_NODE;
  last_active = INVALID_NODE;
  orphan_nodes.clear();
  time = 0;

  for (size_t i = 0; i < nodes.size(); ++i) {
    if (node_blocks[i] == self) {
      Node &n = nodes[i];
      n.next_active = INVALID_NODE;
      n.timestamp = 0;
      if (n.tr_cap != 0) {
        // Node n is connected to the source or sink
        n.is_sink = n.tr_cap < 0;  // Negative capacity goes to sink
        n.parent = TERMINAL_ARC;
        n.dist = 1;
        make_active(i);
      } else {
        n.parent = INVALID_ARC;
      }
    }
  }

  initialized = true;
}

inline void ParallelGraph::GraphBlock::make_active(NodeIdx i) {
  if (nodes[i].next_active == INVALID_NODE) {
    // It's not in the active list yet
    if (last_active != INVALID_NODE) {
      nodes[last_active].next_active = i;
    } else {
      first_active = i;
    }
    last_active = i;
    nodes[i].next_active = i;
  }
}

inline void ParallelGraph::GraphBlock::make_front_orphan(NodeIdx i) {
  nodes[i].parent = ORPHAN_ARC;
  orphan_nodes.push_front(i);
}

inline void ParallelGraph::GraphBlock::make_back_orphan(NodeIdx i) {
  nodes[i].parent = ORPHAN_ARC;
  orphan_nodes.push_back(i);
}

inline NodeIdx ParallelGraph::GraphBlock::next_active() {
  NodeIdx i;
  // Pop nodes from the active list until we find a valid one or run out of
  // nodes
  for (i = first_active; i != INVALID_NODE; i = first_active) {
    // Pop node from active list
    Node &n = nodes[i];
    if (n.next_active == i) {
      // This is the last node in the active list so "clear" it
      first_active = INVALID_NODE;
      last_active = INVALID_NODE;
    } else {
      first_active = n.next_active;
    }
    n.next_active = INVALID_NODE;  // Mark as not active

    if (n.parent != INVALID_ARC) {
      // Valid active node found
      break;
    }
  }
  return i;
}

inline void ParallelGraph::GraphBlock::augment(ArcIdx middle_idx) {
  Arc &middle = arcs[middle_idx];
  Arc &middle_sister = sister(middle_idx);
  // Step 1: Find bottleneck capacity
  Term bottleneck = middle.r_cap;
  bottleneck = std::min(bottleneck, tree_bottleneck(middle_sister.head, true));
  bottleneck = std::min(bottleneck, tree_bottleneck(middle.head, false));

  // Step  2: Augment along source and sink tree
  middle_sister.r_cap += bottleneck;
  middle.r_cap -= bottleneck;
  augment_tree(middle_sister.head, bottleneck, true);
  augment_tree(middle.head, bottleneck, false);

  // Step 3: Add bottleneck to overall flow
  flow += bottleneck;
}

inline Term ParallelGraph::GraphBlock::tree_bottleneck(NodeIdx start,
                                                       bool source_tree) const {
  NodeIdx i = start;
  Term bottleneck = std::numeric_limits<Term>::max();
  while (true) {
    ArcIdx a = nodes[i].parent;
    if (a == TERMINAL_ARC) {
      break;
    }
    Cap r_cap = source_tree ? sister(a).r_cap : arcs[a].r_cap;
    bottleneck = std::min<Term>(bottleneck, r_cap);
    i = arcs[a].head;
  }
  const Term tr_cap = nodes[i].tr_cap;
  return std::min<Term>(bottleneck, source_tree ? tr_cap : -tr_cap);
}

inline void ParallelGraph::GraphBlock::augment_tree(NodeIdx start,
                                                    Term bottleneck,
                                                    bool source_tree) {
  NodeIdx i = start;
  while (true) {
    ArcIdx ai = nodes[i].parent;
    if (ai == TERMINAL_ARC) {
      break;
    }
    Arc &a = source_tree ? arcs[ai] : sister(ai);
    Arc &b = source_tree ? sister(ai) : arcs[ai];
    a.r_cap += bottleneck;
    b.r_cap -= bottleneck;
    if (b.r_cap == 0) {
      make_front_orphan(i);
    }
    i = arcs[ai].head;
  }
  nodes[i].tr_cap += source_tree ? -bottleneck : bottleneck;
  if (nodes[i].tr_cap == 0) {
    make_front_orphan(i);
  }
}

inline ArcIdx ParallelGraph::GraphBlock::grow_search_tree(NodeIdx start) {
  return nodes[start].is_sink ? grow_search_tree_impl<false>(start)
                              : grow_search_tree_impl<true>(start);
}

template <bool source>
inline ArcIdx ParallelGraph::GraphBlock::grow_search_tree_impl(
    NodeIdx start_idx) {
  const Node &start = nodes[start_idx];
  ArcIdx ai;
  // Add neighbor nodes search tree until we find a node from the other search
  // tree or run out of neighbors
  for (ai = start.first; ai != INVALID_ARC; ai = arcs[ai].next) {
    if (sister_or_arc(ai, !source).r_cap > 0) {
      Node &n = head_node(ai);
      if (n.parent == INVALID_ARC) {
        // This node is not yet in a tree so claim it for this one
        n.is_sink = !source;
        n.parent = sister_idx(ai);
        n.timestamp = start.timestamp;
        n.dist = start.dist + 1;
        make_active(arcs[ai].head);
      } else if (n.is_sink == source) {
        // Found a node from the other search tree so abort
        if (!source) {
          // If we are growing the sink tree we instead return the sister arc
          ai = sister_idx(ai);
        }
        break;
      } else if (n.timestamp <= start.timestamp && n.dist > start.dist) {
        // Heuristic: trying to make the distance from j to the source/sink
        // shorter
        n.parent = sister_idx(ai);
        n.timestamp = n.timestamp;
        n.dist = start.dist + 1;
      }
    }
  }
  return ai;
}

inline void ParallelGraph::GraphBlock::process_orphan(NodeIdx i) {
  if (nodes[i].is_sink) {
    process_orphan_impl<false>(i);
  } else {
    process_orphan_impl<true>(i);
  }
}

template <bool source>
inline void ParallelGraph::GraphBlock::process_orphan_impl(NodeIdx i) {
  Node &n = nodes[i];
  static const int32_t INF_DIST = std::numeric_limits<int32_t>::max();
  int32_t min_d = INF_DIST;
  ArcIdx min_a0 = INVALID_ARC;
  // Try to find a new parent
  for (ArcIdx a0 = n.first; a0 != INVALID_ARC; a0 = arcs[a0].next) {
    if (sister_or_arc(a0, source).r_cap > 0) {
      NodeIdx j = arcs[a0].head;
      ArcIdx a = nodes[j].parent;
      if (nodes[j].is_sink != source && a != INVALID_ARC) {
        // Check origin of m
        int32_t d = 0;
        while (true) {
          Node &m = nodes[j];
          if (m.timestamp == time) {
            d += m.dist;
            break;
          }
          a = m.parent;
          d++;
          if (a == TERMINAL_ARC) {
            m.timestamp = time;
            m.dist = 1;
            break;
          }
          if (a == ORPHAN_ARC) {
            d = INF_DIST;  // infinite distance
            break;
          }
          j = arcs[a].head;
        }
        if (d < INF_DIST) {
          // m originates from the source
          if (d < min_d) {
            min_a0 = a0;
            min_d = d;
          }
          // Set marks along the path
          j = arcs[a0].head;
          while (nodes[j].timestamp != time) {
            Node &m = nodes[j];
            m.timestamp = time;
            m.dist = d--;
            j = arcs[m.parent].head;
          }
        }
      }
    }
  }
  n.parent = min_a0;
  if (min_a0 != INVALID_ARC) {
    n.timestamp = time;
    n.dist = min_d + 1;
  } else {
    // No parent was found so process neighbors
    for (ArcIdx a0 = n.first; a0 != INVALID_ARC; a0 = arcs[a0].next) {
      if (arcs[a0].r_cap != INACTIVE_ARC) {
        NodeIdx j = arcs[a0].head;
        Node &m = nodes[j];
        if (m.is_sink != source && m.parent != INVALID_ARC) {
          if (sister_or_arc(a0, source).r_cap > 0) {
            make_active(j);
          }
          ArcIdx pa = m.parent;
          if (pa != TERMINAL_ARC && pa != ORPHAN_ARC && arcs[pa].head == i) {
            make_back_orphan(j);
          }
        }
      }
    }
  }
}

inline bool ParallelGraph::should_activate(NodeIdx i, NodeIdx j) {
  Node &ni = nodes[i];
  Node &nj = nodes[j];
  // If one of the nodes were previously visited, but the other wasn't,
  // or they are different (source/sink) and have both previously been visited.
  return (ni.parent == INVALID_ARC && nj.parent != INVALID_ARC) ||
         (ni.parent != INVALID_ARC && nj.parent == INVALID_ARC) ||
         (ni.parent != INVALID_ARC && nj.parent != INVALID_ARC &&
          ni.is_sink != nj.is_sink);
}

ParallelGraph read_max_file(int &n, int &m, int &s, int &t) {
  std::string line;
  while (getline(std::cin, line)) {
    std::stringstream ss(line);
    char type;
    if (!(ss >> type)) continue;
    if (type == 'c') continue;
    if (type == 'p') {
      std::string tmp;
      ss >> tmp >> n >> m;
      break;
    }
  }
  const int num_blocks = std::min(n, 100);
  ParallelGraph graph(n, m, num_blocks);
  for (size_t i = 0; i < num_blocks; i++) {
    if (i == num_blocks - 1) {
      graph.add_node(n / num_blocks + n % num_blocks, i);
    } else {
      graph.add_node(n / num_blocks, i);
    }
  }
  while (getline(std::cin, line)) {
    std::stringstream ss(line);
    char type;
    if (!(ss >> type)) continue;
    if (type == 'c') continue;
    if (type == 'n') {
      int tmp;
      char label;
      ss >> tmp >> label;
      if (label == 's')
        s = tmp;
      else
        t = tmp;
      continue;
    }
    if (type == 'a') {
      int u, v, cap;
      ss >> u >> v >> cap;
      if (u == v) continue;
      u--;
      v--;
      graph.add_edge(u, v, cap, 0);
      // debug("add_edge", u, v, cap);
      continue;
    }
  }
  s--;
  t--;
  graph.add_tweights(s, Term(1e18), 0);
  graph.add_tweights(t, 0, Term(1e18));
  return graph;
}
ParallelGraph read_input_luogu(int &n, int &m, int &s, int &t) {
  std::cin >> n >> m >> s >> t;
  const int num_blocks = std::min(n, 100);
  ParallelGraph graph(n, m, num_blocks);
  for (size_t i = 0; i < num_blocks; i++) {
    if (i == num_blocks - 1) {
      graph.add_node(n / num_blocks + n % num_blocks, i);
    } else {
      graph.add_node(n / num_blocks, i);
    }
  }
  s--;
  t--;
  for (int i = 0; i < m; i++) {
    int u, v, c;
    std::cin >> u >> v >> c;
    u--;
    v--;
    if (u == v) continue;

    graph.add_edge(u, v, c, 0);
  }
  graph.add_tweights(s, Term(1e18), 0);
  graph.add_tweights(t, 0, Term(1e18));
  return graph;
}
ParallelGraph read_input_aizu(int &n, int &m, int &s, int &t) {
  std::cin >> n >> m;
  s = 0;
  t = n - 1;
  const int num_blocks = std::min(n, 100);
  ParallelGraph graph(n, m, num_blocks);
  for (size_t i = 0; i < num_blocks; i++) {
    if (i == num_blocks - 1) {
      graph.add_node(n / num_blocks + n % num_blocks, i);
    } else {
      graph.add_node(n / num_blocks, i);
    }
  }
  for (int i = 0; i < m; i++) {
    int u, v, c;
    std::cin >> u >> v >> c;
    if (u == v) continue;

    graph.add_edge(u, v, c, 0);
  }
  graph.add_tweights(s, Term(1e18), 0);
  graph.add_tweights(t, 0, Term(1e18));
  return graph;
}

int main(int argc, char **argv) {
  std::ios::sync_with_stdio(0);
  std::cin.tie(0);
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <num_threads>\n";
  }
  num_threads = std::stoi(argv[1]);

  int n, m, s, t;
  auto graph = read_max_file(n, m, s, t);

  if (n == 1) {
    std::cout << "0\n";
    return 0;
  }

  std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();
  std::cout << graph.maxflow() << '\n';
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cerr << "Time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
                   .count()
            << "ms\n";
  return 0;
}