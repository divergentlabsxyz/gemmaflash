// Native expert gate for SSD-streamed MoE decode.
//
// Each MoE layer keeps C of its experts resident in slot tensors. Decode used
// to sync Python on every layer's router output to look up / read the routed
// experts. Here the lookup happens inside the GPU command stream instead:
//
//   gate(layer, idx, pidx) -> slot indices (uint32, shape of idx)
//
// is an MLX primitive whose eval_gpu encodes MLX's fast-fence kernels:
//   input_coherent(idx)          -- router output visible to the CPU
//   fence_update(flag0 = 2n+1)   -- gate n is ready
//   fence_wait(flag1 >= 2n+2)    -- one GPU thread spins until the slots are in
// and queues job n for a native worker thread. (An MTLSharedEvent
// signal/wait pair costs ~130 us per round trip; spinning on coherent(system)
// memory on both sides avoids it.) The worker spins on flag0, reads idx from
// unified memory, does the LRU lookup, preads misses
// straight into the slot tensors (F_NOCACHE), writes the slot indices into the
// primitive's output buffer and signals 2n+2. Then it starts the next layer's
// prefetch. Python builds and submits the whole forward without blocking.
//
// The GPU is stopped at the gate while the worker runs, after every earlier
// command has finished, so the worker may overwrite any slot of this layer
// that the current gate doesn't route to; the next layer's slots are not read
// again until its own gate.

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "mlx/allocator.h"
#include "mlx/array.h"
#include "mlx/backend/metal/device.h"
#include "mlx/primitives.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"
#include "mlx/utils.h"

namespace nb = nanobind;
using namespace nb::literals;
namespace mx = mlx::core;
using clk = std::chrono::steady_clock;

namespace {

double secs(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

// ---------------------------------------------------------------- thread pool

class Pool {
 public:
  explicit Pool(int n) {
    for (int i = 0; i < n; i++) {
      std::thread([this] { run(); }).detach();
    }
  }
  std::shared_future<void> submit(std::function<void()> f) {
    auto t = std::make_shared<std::packaged_task<void()>>(std::move(f));
    auto fut = t->get_future().share();
    {
      std::lock_guard lk(mu_);
      q_.push_back([t] { (*t)(); });
    }
    cv_.notify_one();
    return fut;
  }

 private:
  void run() {
    for (;;) {
      std::function<void()> f;
      {
        std::unique_lock lk(mu_);
        cv_.wait(lk, [this] { return !q_.empty(); });
        f = std::move(q_.front());
        q_.pop_front();
      }
      f();
    }
  }
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::function<void()>> q_;
};

// ---------------------------------------------------------------- layer state

struct Field {
  uint8_t* base;  // slot tensor memory (unified, written by the CPU)
  size_t nbytes;  // bytes of one expert row
  size_t offset;  // offset of the field in an expert record
};

struct Layer {
  int fd = -1;
  size_t record = 0;
  int n = 0, cap = 0;
  std::vector<mx::array> arrays;  // keeps the slot tensors alive
  std::vector<Field> fields;
  std::vector<int> slot_of;
  // LRU over resident experts, least recent first; sentinel = n.
  std::vector<int> prv, nxt;
  std::vector<char> in_lru;
  std::map<int, std::shared_future<void>> pending;  // background reads
  std::vector<int> prefetched;
  int next = -1;  // layer predicted by this layer's router (prefetch target)

  void lru_init() {
    prv.assign(n + 1, n);
    nxt.assign(n + 1, n);
    in_lru.assign(n, 0);
  }
  void lru_remove(int e) {
    nxt[prv[e]] = nxt[e];
    prv[nxt[e]] = prv[e];
    in_lru[e] = 0;
  }
  void lru_push_back(int e) {
    if (in_lru[e]) lru_remove(e);
    int last = prv[n];
    nxt[last] = e;
    prv[e] = last;
    nxt[e] = n;
    prv[n] = e;
    in_lru[e] = 1;
  }
  void lru_push_front(int e) {
    if (in_lru[e]) lru_remove(e);
    int first = nxt[n];
    prv[first] = e;
    nxt[e] = first;
    prv[e] = n;
    nxt[n] = e;
    in_lru[e] = 1;
  }
  // One slot per expert in need, never evicting an expert in keep.
  std::vector<int> evict_for(size_t count, const std::unordered_set<int>& keep) {
    std::vector<int> slots;
    int e = nxt[n];
    while (slots.size() < count) {
      if (e == n) throw std::runtime_error("xgate: no evictable slot");
      int nx = nxt[e];
      if (!keep.count(e)) {
        slots.push_back(slot_of[e]);
        lru_remove(e);
        slot_of[e] = -1;
      }
      e = nx;
    }
    return slots;
  }
  void wait(const std::vector<int>* experts, double& wait_s) {
    auto t = clk::now();
    bool waited = false;
    if (experts == nullptr) {
      for (auto& [e, f] : pending) {
        f.wait();
        waited = true;
      }
      pending.clear();
    } else {
      for (int e : *experts) {
        auto it = pending.find(e);
        if (it != pending.end()) {
          it->second.wait();
          pending.erase(it);
          waited = true;
        }
      }
    }
    if (waited) wait_s += secs(t, clk::now());
  }
};

thread_local std::vector<uint8_t>* tl_stage = nullptr;

void read_one(Layer& L, int e, int row) {
  // One aligned read of the whole record, then copy the fields out: uncached
  // reads straight into the fields' unaligned rows are much slower.
  if (tl_stage == nullptr) {
    tl_stage = new std::vector<uint8_t>(L.record + 16384);
  }
  if (tl_stage->size() < L.record + 16384) tl_stage->resize(L.record + 16384);
  auto* stage = reinterpret_cast<uint8_t*>(
      (reinterpret_cast<uintptr_t>(tl_stage->data()) + 16383) & ~uintptr_t(16383));
  size_t got = 0;
  while (got < L.record) {
    ssize_t r = pread(L.fd, stage + got, L.record - got, off_t(e) * L.record + got);
    if (r <= 0) throw std::runtime_error("xgate: short expert read");
    got += size_t(r);
  }
  for (auto& f : L.fields) {
    std::memcpy(f.base + size_t(row) * f.nbytes, stage + f.offset, f.nbytes);
  }
}

struct Stats {
  uint64_t lookups = 0, misses = 0, gates = 0, prefetch_reads = 0, prefetch_used = 0;
  double read_s = 0, hold_s = 0, spin_s = 0, prefetch_wait_s = 0, prefetch_s = 0;
};

struct Job {
  int layer;
  uint64_t n;
  mx::array idx, out;
  std::optional<mx::array> pidx;
};

struct State;

// One per MLX stream: its own flag words, gate counter, job queue and
// worker, so the gates of a stream are served in that stream's order.
struct Channel {
  MTL::Buffer* flags = nullptr;  // u32 at 0: GPU -> CPU, u32 at 128: CPU -> GPU
  std::atomic_uint* ready = nullptr;
  std::atomic_uint* release = nullptr;
  uint64_t counter = 0;  // gates encoded
  std::mutex qmu;
  std::condition_variable qcv, idle_cv;
  std::deque<Job> q;
  bool busy = false;

  void run(State& st);
  void drain() {
    std::unique_lock lk(qmu);
    idle_cv.wait(lk, [&] { return q.empty() && !busy; });
  }
};

struct State {
  std::vector<Layer> layers;
  std::mutex mu;  // layer state: worker job processing and Python calls
  Pool* pool = nullptr;
  int prefetch = 0;
  Stats st;
  std::exception_ptr error;

  std::mutex cmu;
  std::map<int, Channel*> channels;  // by stream index; never freed

  Channel& channel(const mx::Stream& s) {
    std::lock_guard lk(cmu);
    auto& c = channels[s.index];
    if (c == nullptr) {
      c = new Channel();
      c->flags = mx::metal::device(s.device).mtl_device()->newBuffer(
          256, MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked);
      auto* base = static_cast<uint8_t*>(c->flags->contents());
      std::memset(base, 0, 256);
      c->ready = reinterpret_cast<std::atomic_uint*>(base);
      c->release = reinterpret_cast<std::atomic_uint*>(base + 128);
      Channel* cp = c;
      std::thread([this, cp] { cp->run(*this); }).detach();
    }
    return *c;
  }

  void drain() {
    std::vector<Channel*> cs;
    {
      std::lock_guard lk(cmu);
      for (auto& [i, c] : channels) cs.push_back(c);
    }
    for (auto* c : cs) c->drain();
  }

  void process(Job& job, Channel& ch) {
    uint32_t sig = uint32_t(2 * job.n + 1), rel = uint32_t(2 * job.n + 2);
    auto t0 = clk::now();
    for (int spins = 0; ch.ready->load(std::memory_order_acquire) < sig; spins++) {
      if (spins > 4096) std::this_thread::yield();
    }
    auto t1 = clk::now();
    {
      std::lock_guard lk(mu);
      st.spin_s += secs(t0, t1);
      try {
        gate(job);
      } catch (...) {
        if (!error) error = std::current_exception();
      }
      std::atomic_thread_fence(std::memory_order_seq_cst);
      ch.release->store(rel, std::memory_order_release);
      auto t2 = clk::now();
      st.hold_s += secs(t1, t2);
      st.gates++;
      if (job.pidx && prefetch > 0 && layers[job.layer].next >= 0) {
        try {
          prefetch_next(layers[layers[job.layer].next], *job.pidx);
        } catch (...) {
          if (!error) error = std::current_exception();
        }
        st.prefetch_s += secs(t2, clk::now());
      }
    }
  }

  static std::vector<uint32_t> read_indices(const mx::array& a) {
    if (!a.flags().row_contiguous) throw std::runtime_error("xgate: indices must be contiguous");
    std::vector<uint32_t> v(a.size());
    if (a.dtype() == mx::uint32 || a.dtype() == mx::int32) {
      std::memcpy(v.data(), a.data<uint32_t>(), a.size() * 4);
    } else {
      throw std::runtime_error("xgate: indices must be (u)int32");
    }
    return v;
  }

  void read_many(Layer& L, const std::vector<std::pair<int, int>>& jobs) {
    if (jobs.empty()) return;
    auto t = clk::now();
    if (jobs.size() == 1) {
      read_one(L, jobs[0].first, jobs[0].second);
    } else {
      std::vector<std::shared_future<void>> fs;
      for (auto [e, r] : jobs) fs.push_back(pool->submit([&L, e = e, r = r] { read_one(L, e, r); }));
      for (auto& f : fs) f.get();
    }
    st.read_s += secs(t, clk::now());
  }

  void gate(Job& job) {
    Layer& L = layers[job.layer];
    auto idx = read_indices(job.idx);
    std::vector<int> used;
    std::unordered_set<int> used_set;
    for (uint32_t e : idx) {
      if (int(e) >= L.n) throw std::runtime_error("xgate: expert index out of range");
      if (used_set.insert(int(e)).second) used.push_back(int(e));
    }
    // Wrong guesses may still be reading: the kernels won't touch them, and
    // they stay out of eviction until done.
    L.wait(&used, st.prefetch_wait_s);
    for (auto it = L.pending.begin(); it != L.pending.end();) {
      if (it->second.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
        it = L.pending.erase(it);
      else
        ++it;
    }
    if (!L.prefetched.empty()) {
      for (int e : L.prefetched) st.prefetch_used += used_set.count(e);
      L.prefetched.clear();
    }
    std::vector<int> miss;
    for (int e : used)
      if (L.slot_of[e] < 0) miss.push_back(e);
    st.lookups += used.size();
    st.misses += miss.size();
    if (!miss.empty()) {
      std::unordered_set<int> keep = used_set;
      for (auto& [e, f] : L.pending) keep.insert(e);
      auto slots = L.evict_for(miss.size(), keep);
      std::vector<std::pair<int, int>> jobs;
      for (size_t i = 0; i < miss.size(); i++) jobs.push_back({miss[i], slots[i]});
      read_many(L, jobs);
      for (size_t i = 0; i < miss.size(); i++) {
        L.slot_of[miss[i]] = slots[i];
        L.lru_push_back(miss[i]);
      }
    }
    for (int e : used) L.lru_push_back(e);
    auto* out = job.out.data<uint32_t>();
    for (size_t i = 0; i < idx.size(); i++) out[i] = uint32_t(L.slot_of[idx[i]]);
  }

  // Start reading the most-routed predicted experts that aren't resident. They
  // take the least recent slots and are filed as least recent themselves, so
  // a wrong guess is the first thing evicted.
  void prefetch_next(Layer& N, const mx::array& pidx) {
    N.wait(nullptr, st.prefetch_wait_s);
    auto pred = read_indices(pidx);
    std::map<int, int> count;
    for (uint32_t e : pred) count[int(e)]++;
    std::vector<std::pair<int, int>> order(count.begin(), count.end());  // ascending expert
    std::stable_sort(order.begin(), order.end(),
                     [](auto& a, auto& b) { return a.second > b.second; });
    std::vector<int> cand;
    for (auto& [e, c] : order) {
      if (int(cand.size()) >= prefetch) break;
      if (N.slot_of[e] < 0) cand.push_back(e);
    }
    if (cand.empty()) return;
    auto slots = N.evict_for(cand.size(), {});
    for (size_t i = 0; i < cand.size(); i++) {
      int e = cand[i], s = slots[i];
      N.slot_of[e] = s;
      N.lru_push_front(e);
      Layer* np = &N;
      N.pending[e] = pool->submit([np, e, s] { read_one(*np, e, s); });
    }
    N.prefetched = cand;
    st.prefetch_reads += cand.size();
  }

  void check() {
    if (error) {
      auto e = error;
      error = nullptr;
      std::rethrow_exception(e);
    }
  }

};

void Channel::run(State& st) {
  for (;;) {
    Job job = [&] {
      std::unique_lock lk(qmu);
      qcv.wait(lk, [&] { return !q.empty(); });
      busy = true;
      Job j = std::move(q.front());
      q.pop_front();
      return j;
    }();
    st.process(job, *this);
    {
      std::lock_guard lk(qmu);
      busy = false;
    }
    idle_cv.notify_all();
  }
}

State& S() {
  static State* s = new State();  // never destroyed: the workers are detached
  return *s;
}

// ---------------------------------------------------------------- primitive

class ExpertGate : public mx::Primitive {
 public:
  ExpertGate(mx::Stream s, int layer) : mx::Primitive(s), layer_(layer) {}

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("ExpertGate is GPU-only");
  }

  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    auto& st = S();
    auto& out = outputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto s = stream();
    auto& ch = st.channel(s);
    uint64_t n = ch.counter++;
    uint32_t sig = uint32_t(2 * n + 1), rel = uint32_t(2 * n + 2);
    auto& d = mx::metal::device(s.device);
    auto& enc = mx::metal::get_command_encoder(s);
    // Router outputs written through to memory the CPU sees.
    auto* coh = d.get_kernel("input_coherent");
    for (auto& a : inputs) {
      uint32_t nthreads = uint32_t((a.nbytes() + 3) / 4);
      enc.set_compute_pipeline_state(coh);
      enc.set_input_array(a, 0);
      enc.set_bytes(nthreads, 1);
      enc.dispatch_threadgroups(MTL::Size((nthreads + 1023) / 1024, 1, 1), MTL::Size(1024, 1, 1));
    }
    enc.barrier();
    enc.set_compute_pipeline_state(d.get_kernel("fence_update"));
    enc.set_buffer(ch.flags, 0, 0);
    enc.set_bytes(sig, 1);
    enc.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
    enc.end_encoding();
    // Kernels reading the slot indices start only after the spin is over.
    enc.register_output_array(out);
    enc.set_compute_pipeline_state(d.get_kernel("fence_wait"));
    enc.set_buffer(ch.flags, 0, 128);
    enc.set_bytes(rel, 1);
    enc.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
    enc.end_encoding(true);
    {
      std::lock_guard lk(ch.qmu);
      Job j{layer_, n, inputs[0], out, std::nullopt};
      if (inputs.size() > 1) j.pidx = inputs[1];
      ch.q.push_back(std::move(j));
    }
    ch.qcv.notify_one();
  }

  const char* name() const override {
    return "ExpertGate";
  }

 private:
  int layer_;
};

// ---------------------------------------------------------------- bindings

int add_layer(const std::string& path, size_t record, int num_experts, int capacity,
              std::vector<size_t> offsets, std::vector<size_t> nbytes, std::vector<mx::array> arrays) {
  auto& st = S();
  std::lock_guard lk(st.mu);
  if (st.pool == nullptr) st.pool = new Pool(4);
  Layer L;
  L.fd = open(path.c_str(), O_RDONLY);
  if (L.fd < 0) throw std::runtime_error("xgate: cannot open " + path);
  fcntl(L.fd, F_NOCACHE, 1);
  L.record = record;
  L.n = num_experts;
  L.cap = capacity;
  for (auto& a : arrays) mx::eval(a);
  for (size_t i = 0; i < arrays.size(); i++) {
    auto& a = arrays[i];
    if (!a.flags().row_contiguous || a.shape(0) != capacity || a.nbytes() != nbytes[i] * capacity)
      throw std::runtime_error("xgate: slot tensor layout mismatch");
    L.fields.push_back({a.data<uint8_t>(), nbytes[i], offsets[i]});
  }
  L.arrays = std::move(arrays);
  L.slot_of.assign(num_experts, -1);
  L.lru_init();
  st.layers.push_back(std::move(L));
  Layer& R = st.layers.back();
  std::vector<std::pair<int, int>> jobs;
  for (int e = 0; e < capacity; e++) jobs.push_back({e, e});
  st.read_many(R, jobs);
  for (int e = 0; e < capacity; e++) {
    R.slot_of[e] = e;
    R.lru_push_back(e);
  }
  return int(st.layers.size()) - 1;
}

mx::array gate(int layer, const mx::array& idx, std::optional<mx::array> pidx) {
  auto s = mx::default_stream(mx::Device::gpu);
  std::vector<mx::array> inputs{idx};
  if (pidx) inputs.push_back(*pidx);
  return mx::array(idx.shape(), mx::uint32, std::make_shared<ExpertGate>(s, layer), std::move(inputs));
}

}  // namespace

NB_MODULE(_xgate, m) {
  nb::module_::import_("mlx.core");

  m.def("add_layer", &add_layer, "path"_a, "record"_a, "num_experts"_a, "capacity"_a, "offsets"_a,
        "nbytes"_a, "arrays"_a, "Register a streamed MoE layer; reads its first `capacity` experts.");
  m.def("set_next", [](int layer, int next) { S().layers.at(layer).next = next; });
  m.def("set_prefetch", [](int n) { S().prefetch = n; });
  m.def("gate", &gate, "layer"_a, "idx"_a, "pidx"_a = nb::none(),
        "Slot indices for idx, resolved on the worker while the GPU waits in-stream.");
  m.def(
      "drain",
      [] {
        S().drain();
        std::lock_guard lk(S().mu);
        S().check();
      },
      nb::call_guard<nb::gil_scoped_release>());

  // Shared LRU state for the Python prefill path (call drain() first).
  m.def("slot_of", [](int l) { std::lock_guard lk(S().mu); return S().layers.at(l).slot_of; });
  m.def(
      "wait_all",
      [](int l) {
        auto& st = S();
        std::lock_guard lk(st.mu);
        st.layers.at(l).wait(nullptr, st.st.prefetch_wait_s);
        st.layers.at(l).prefetched.clear();
      },
      nb::call_guard<nb::gil_scoped_release>());
  m.def("evict_for", [](int l, size_t count, std::vector<int> keep) {
    std::lock_guard lk(S().mu);
    return S().layers.at(l).evict_for(count, std::unordered_set<int>(keep.begin(), keep.end()));
  });
  m.def("assign", [](int l, std::vector<int> experts, std::vector<int> slots) {
    std::lock_guard lk(S().mu);
    auto& L = S().layers.at(l);
    for (size_t i = 0; i < experts.size(); i++) {
      L.slot_of[experts[i]] = slots[i];
      L.lru_push_back(experts[i]);
    }
  });
  m.def("touch", [](int l, std::vector<int> experts) {
    std::lock_guard lk(S().mu);
    auto& L = S().layers.at(l);
    for (int e : experts) L.lru_push_back(e);
  });
  m.def("stats", [] {
    std::lock_guard lk(S().mu);
    auto& s = S().st;
    nb::dict d;
    d["decode_lookups"] = s.lookups;
    d["decode_misses"] = s.misses;
    d["gates"] = s.gates;
    d["read_s"] = s.read_s;
    d["hold_s"] = s.hold_s;
    d["spin_s"] = s.spin_s;
    d["prefetch_reads"] = s.prefetch_reads;
    d["prefetch_used"] = s.prefetch_used;
    d["prefetch_wait_s"] = s.prefetch_wait_s;
    d["prefetch_s"] = s.prefetch_s;
    return d;
  });
  m.def("reset_stats", [] {
    std::lock_guard lk(S().mu);
    S().st = Stats{};
  });
}
