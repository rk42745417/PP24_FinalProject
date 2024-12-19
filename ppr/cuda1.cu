#include <bits/stdc++.h>
#include <cooperative_groups.h>

using namespace std;
using namespace cooperative_groups;

#define StarBurstStream ios_base::sync_with_stdio(false); cin.tie(0);
#define iter(a) a.begin(), a.end()
#define pb emplace_back
#define ff first
#define ss second
#define SZ(a) int(a.size())

using ll = int;
using pii = pair<int, int>;
using pll = pair<ll, ll>;
#ifdef zisk
void debug(){cerr << "\n";}
template<class T, class ... U> void debug(T a, U ... b){cerr << a << " ", debug(b...);}
template<class T> void pary(T l, T r) {
	while (l != r) cerr << *l << " ", l++;
	cerr << "\n";
}
#else
#define debug(...) void()
#define pary(...) void()
#endif

#define CHECK(x) { \
    cudaError_t status = x; \
    if (status != cudaSuccess) { \
        std::cout << "Error: (" << __LINE__ << ") " << cudaGetErrorString(status) << std::endl; \
        exit(1); \
    }  \
}
#define CHECK_LAST() { \
    cudaError_t status = cudaGetLastError(); \
    if (status != cudaSuccess) { \
        std::cout << "Error: (" << __LINE__ << ") " << cudaGetErrorString(status) << std::endl; \
        exit(1); \
    }  \
}

template<typename A, typename B>
ostream& operator<<(ostream& o, pair<A, B> p){
    return o << '(' << p.ff << ',' << p.ss << ')';
}

template <class T>
struct edge {
    int from, to, rev;
    T cap, flow;
};

template <class T>
struct HostFlowNetwork {
    int n, m = 0;
    vector<vector<edge<T>>> g;
    HostFlowNetwork(int _n): n(_n), g(n) {}
    void add_edge(int u, int v, T cap) {
        g[u].pb(edge<T>({u, v, SZ(g[v]), cap, T(0)}));
        g[v].pb(edge<T>({v, u, SZ(g[u]) - 1, T(0), T(0)}));
        m++;
    }
};

template <class T>
struct PreflowPushRelabel {
    int n, m, s, t;
    edge<T> *pool;
    edge<T> **g;
    T *excess;
    int *num_edges;
    int *h;
    PreflowPushRelabel(int _n, int _s, int _t, HostFlowNetwork<T> &src): n(_n), m(src.m), s(_s), t(_t) {
        CHECK(cudaMalloc(&pool, sizeof(edge<T>) * 2 * m));
        CHECK(cudaMalloc(&g, sizeof(edge<T>*) * n));
        CHECK(cudaMalloc(&excess, sizeof(T) * n));
        CHECK(cudaMalloc(&num_edges, sizeof(int) * n));
        CHECK(cudaMalloc(&h, sizeof(int) * n));
        edge<T> *ptr = pool;
        vector<edge<T>*> host_g(n);
        vector<T> host_excess(n);
        vector<int> host_num_edges(n);
        vector<int> host_h(n);
        vector<edge<T>> host_pool;
        for (auto &e : src.g[s]) {
            e.flow += e.cap;
            src.g[e.to][e.rev].flow -= e.cap;
            host_excess[e.from] -= e.cap;
            host_excess[e.to] += e.cap;
        }
        for (int i = 0; i < n; i++) {
            host_g[i] = ptr;
            for (auto &e : src.g[i]) host_pool.pb(e);
            ptr += src.g[i].size();
            host_num_edges[i] = src.g[i].size();
        }
        host_h[s] = n;

        CHECK(cudaMemcpy(pool, host_pool.data(), sizeof(edge<T>) * 2 * m, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(g, host_g.data(), sizeof(edge<T>*) * n, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(excess, host_excess.data(), sizeof(T) * n, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(num_edges, host_num_edges.data(), sizeof(int) * n, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(h, host_h.data(), sizeof(int) * n, cudaMemcpyHostToDevice));
    }
    HostFlowNetwork<T> hostData() {
        HostFlowNetwork<T> ret(n);
        vector<edge<T>> host_pool(2 * m);
        CHECK(cudaMemcpy(host_pool.data(), pool, sizeof(edge<T>) * 2 * m, cudaMemcpyDeviceToHost));
        ret.m = m;
        for (auto &e : host_pool) {
            ret.g[e.from].pb(e);
        }
        return ret;
    }
    __device__
    void add_flow(edge<T> &e, T f) {
        e.flow += f;
        g[e.to][e.rev].flow -= f;
        atomicAdd(&excess[e.from], -f);
        atomicAdd(&excess[e.to], f);
    }
    __device__
    bool push(edge<T> &e) {
        if (excess[e.from] <= 0 || e.cap - e.flow <= 0 || h[e.from] != h[e.to] + 1)
            return false;
        T f = min(excess[e.from], e.cap - e.flow);
        add_flow(e, f);
        return true;
    }
    __device__
    bool relabel(int u) {
        if (excess[u] <= 0) return false;
        int min_h = INT_MAX;
        for (int i = 0; i < num_edges[u]; i++) {
            auto &e = g[u][i];
            if (e.flow < e.cap) {
                min_h = min(min_h, h[e.to]);
            }
        }
        if (min_h == INT_MAX || min_h < h[u]) return false;
        h[u] = min_h + 1;
        return true;
    }
    __device__
    void solve(T *flow, int *upd) {
        grid_group grp = this_grid();
        int processSize = (n + gridDim.x * blockDim.x - 1) / (gridDim.x * blockDim.x);
        int id = grp.thread_rank();
        int l = processSize * id, r = l + processSize;
        for (int k = 0; ; k++) {
            bool ok = false;
            for (int i = l; i < n && i < r; i++) {
                if (i == t || excess[i] <= 0) continue;
                for (int j = 0; j < num_edges[i]; j++) {
                    auto &e = g[i][j];
                    if (push(e)) ok = true;
                }
            }
            grp.sync();
            for (int i = l; i < n && i < r; i++) {
                if (i == t || excess[i] <= 0) continue;
                if (i != t && relabel(i)) ok = true;
            }
            if (ok) *upd = k;
            grp.sync();
            if (id == 0 && k % 1000 == 0) printf("done %d %d\n", k, excess[t]);
            if (*upd != k) break;
        }
        if (id == 0)
            *flow = excess[t];
    }
};

template<class T>
__global__
void call_solve(PreflowPushRelabel<T> ppr, T *flow, int *upd) {
    ppr.solve(flow, upd);
}

template<class T>
T solve(PreflowPushRelabel<T> ppr) {
    T *flow;
    int *upd;
    int _upd = -1;
    CHECK(cudaMalloc(&flow, sizeof(T)));
    CHECK(cudaMalloc(&upd, sizeof(int)));
    CHECK(cudaMemcpy(upd, &_upd, sizeof(int), cudaMemcpyHostToDevice));
    void *param[] = {&ppr, &flow, &upd};
    cudaLaunchCooperativeKernel(call_solve<T>, 128, 1024, param);
    CHECK_LAST();
    CHECK(cudaDeviceSynchronize());
    T ans;
    CHECK(cudaMemcpy(&ans, flow, sizeof(T), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(flow));
    CHECK(cudaFree(upd));
    return ans;
}

HostFlowNetwork<ll> read_input(int &n, int &m, int &s, int &t) {
    cin >> n >> m >> s >> t;
    s--; t--;
    HostFlowNetwork<ll> flow(n);
    for (int i = 0; i < m; i++) {
        int u, v, cap;
        cin >> u >> v >> cap;
        u--; v--;
        flow.add_edge(u, v, cap);
    }
    return flow;
}

HostFlowNetwork<ll> read_max_file(int &n, int &m, int &s, int &t) {
    string line;
    while (getline(cin, line)) {
        stringstream ss(line);
        char type;
        if (!(ss >> type)) continue;
        if (type == 'c') continue;
        if (type == 'p') {
            string tmp;
            ss >> tmp >> n >> m;
            break;
        }
    }
    HostFlowNetwork<ll> flow(n);
    int cnt = 0;
    while (getline(cin, line)) {
        stringstream ss(line);
        char type;
        if (!(ss >> type)) continue;
        if (type == 'c') continue;
        if (type == 'n') {
            int tmp; char label;
            ss >> tmp >> label;
            if (label == 's') s = tmp;
            else t = tmp;
            continue;
        }
        if (type == 'a') {
            cnt++;
            int u, v, cap;
            ss >> u >> v >> cap;
            if (cnt % 10000 == 0) debug("read", cnt);
            u--; v--;
            flow.add_edge(u, v, cap);
            //debug("add_edge", u, v, cap);
            continue;
        }
    }
    s--; t--;
    return flow;
}

int main(){
    StarBurstStream;
    cudaSetDevice(1);

    int n, m, s, t;
    //auto input = read_max_file(n, m, s, t);
    auto input = read_input(n, m, s, t);
    debug("read done", m, input.m);
    PreflowPushRelabel flow(n, s, t, input);
    //debug("test", n, m, s, t);
    auto start_time = std::chrono::high_resolution_clock::now();
    cout << solve(flow) << endl;
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;

    std::cout << "Time: " << elapsed_seconds.count() * 1000.0 << " ms" << std::endl;

    // max flow check
    auto ret = flow.hostData();
    for (int i = 0; i < n; i++) {
        if (i == s || i == t) continue;
        ll total = 0;
        for (auto &e : ret.g[i])
            total += e.flow;
        assert(total == 0);
    }
    for (int i = 0; i < n; i++)
        for (auto &e : ret.g[i])
            assert(e.flow <= e.cap);
    queue<int> q;
    q.push(s);
    vector<bool> vst(n);
    vst[s] = true;
    while (!q.empty()) {
        int now = q.front();
        q.pop();
        for (auto &e : ret.g[s]) {
            if (e.cap == e.flow) continue;
            if (vst[e.to]) continue;
            vst[e.to] = true;
            q.push(e.to);
        }
    }
    assert(!vst[t]);

}
