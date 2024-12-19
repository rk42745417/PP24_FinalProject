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

using ll = long long;
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
    int n;
    vector<vector<edge<T>>> g;
    vector<T> excess;
    HostFlowNetwork(int _n): n(_n), g(n), excess(n) {}
    void add_edge(int u, int v, T cap) {
        g[u].pb(edge<T>({u, v, SZ(g[v]), cap, T(0)}));
        g[v].pb(edge<T>({v, u, SZ(g[u]) - 1, T(0), T(0)}));
    }
};

template <class T>
struct FlowNetwork {
    int n;
    edge<T> *pool;
    edge<T> **g;
    T *excess;
    int *num_edges;
    FlowNetwork(int _n, int m, HostFlowNetwork<T> &src): n(_n) {
        CHECK(cudaMalloc(&pool, sizeof(edge<T>) * 2 * m));
        CHECK(cudaMalloc(&g, sizeof(edge<T>*) * n));
        CHECK(cudaMalloc(&excess, sizeof(T) * n));
        CHECK(cudaMalloc(&num_edges, sizeof(int) * n));
        edge<T> *ptr = pool;
        vector<edge<T>*> host_g(n);
        vector<T> host_excess(n);
        vector<int> host_num_edges(n);
        vector<edge<T>> host_pool;
        for (int i = 0; i < n; i++) {
            host_g[i] = ptr;
            //if (i % 10000 == 0) debug("QQ", i, ptr, src.g[i].data());
            for (auto &e : src.g[i]) host_pool.pb(e);
            //CHECK(cudaMemcpy(ptr, src.g[i].data(), sizeof(edge<T>) * src.g[i].size(), cudaMemcpyHostToDevice));
            ptr += src.g[i].size();
            host_num_edges[i] = src.g[i].size();
        }
        CHECK(cudaMemcpy(pool, host_pool.data(), sizeof(edge<T>) * 2 * m, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(g, host_g.data(), sizeof(edge<T>*) * n, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(excess, host_excess.data(), sizeof(T) * n, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(num_edges, host_num_edges.data(), sizeof(int) * n, cudaMemcpyHostToDevice));
        //pary(iter(host_num_edges));
    }
    __device__
    void add_flow(edge<T> &e, T f) {
        using ull = unsigned long long;
        atomicAdd((ull*)&e.flow, (ull)f);
        atomicAdd((ull*)&g[e.to][e.rev].flow, (ull)-f);
        atomicAdd((ull*)&excess[e.from], (ull)-f);
        atomicAdd((ull*)&excess[e.to], (ull)f);
    }
};

template <class T>
struct PreflowPushRelabel : FlowNetwork<T> {
    int *h;
    PreflowPushRelabel(int n, int m, HostFlowNetwork<T> &src): FlowNetwork<T>(n, m, src) {
        vector<int> host_h(n);
        CHECK(cudaMalloc(&h, sizeof(int) * n));
        CHECK(cudaMemcpy(h, host_h.data(), sizeof(int) * n, cudaMemcpyHostToDevice));
        debug("init ok");
    }
    __device__
    bool push(edge<T> &e) {
        if (FlowNetwork<T>::excess[e.from] <= 0 || e.cap - e.flow <= 0 || h[e.from] != h[e.to] + 1)
            return false;
        T f = min(FlowNetwork<T>::excess[e.from], e.cap - e.flow);
        FlowNetwork<T>::add_flow(e, f);
        //printf("push %d %d %ld\n", e.from, e.to, f);
        return true;
    }
    __device__
    bool relabel(int u) {
        if (FlowNetwork<T>::excess[u] <= 0) return false;
        bool has_edge = false;
        for (int i = 0; i < FlowNetwork<T>::num_edges[u]; i++) {
            auto &e = FlowNetwork<T>::g[u][i];
            if (e.flow < e.cap) {
                has_edge = true;
                if (h[u] > h[e.to])
                    return false;
            }
        }
        if (!has_edge) return false;
        int min_h = INT_MAX;
        for (int i = 0; i < FlowNetwork<T>::num_edges[u]; i++) {
            auto &e = FlowNetwork<T>::g[u][i];
            if (e.flow < e.cap) {
                //printf("QQ %d %d\n", e.to, h[e.to]);
                min_h = min(min_h, h[e.to]);
            }
        }
        h[u] = min_h + 1;
        //printf("relabel %d %d\n", u, h[u]);
        return true;
    }
    __device__
    void init(int s) {
        h[s] = FlowNetwork<T>::n;
        //printf("num %d\n", FlowNetwork<T>::num_edges[s]);
        for (int i = 0; i < FlowNetwork<T>::num_edges[s]; i++) {
            auto &e = FlowNetwork<T>::g[s][i];
            //printf("init %d %d %ld %ld\n", i, e.to, e.cap, e.flow);
            if (e.cap == 0) continue;
            FlowNetwork<T>::add_flow(e, e.cap);
        }
    }
    __device__
    void _solve(int s, int t, T *flow, int *upd) {
        grid_group g = this_grid();
        int processSize = (FlowNetwork<T>::n + gridDim.x * blockDim.x - 1) / (gridDim.x * blockDim.x);
        int id = g.thread_rank();
        int l = processSize * id, r = l + processSize;
        for (int k = 0; ; k++) {
            bool ok = false;
            for (int i = l; i < FlowNetwork<T>::n && i < r; i++) {
                if (i == t || FlowNetwork<T>::excess[i] <= 0) continue;
                if (i != t && relabel(i)) ok = true;
                for (int j = 0; j < FlowNetwork<T>::num_edges[i]; j++) {
                    auto &e = FlowNetwork<T>::g[i][j];
                    if (push(e)) ok = true;
                }
            }
            //if (ok && k % 10000 == 0) printf("ok %d %d\n", k, id);
            if (ok) *upd = k;
            if (id == 0 && k % 10000 == 0) printf("done %d %ld %d\n", k, FlowNetwork<T>::excess[t], h[2]);
            g.sync();
            if (*upd != k) break;
        }
        if (id == 0)
            *flow = FlowNetwork<T>::excess[t];
    }
};

template<class T>
__global__
void call_init(PreflowPushRelabel<T> ppr, int s) {
    ppr.init(s);
}
template<class T>
__global__
void call_solve(PreflowPushRelabel<T> ppr, int s, int t, T *flow, int *upd) {
    ppr._solve(s, t, flow, upd);
}

template<class T>
T solve(PreflowPushRelabel<T> ppr, int s, int t) {
    call_init<<<1, 1>>>(ppr, s);
    CHECK_LAST();
    CHECK(cudaDeviceSynchronize());
    T *flow;
    int *upd;
    int _upd = -1;
    CHECK(cudaMalloc(&flow, sizeof(T)));
    CHECK(cudaMalloc(&upd, sizeof(int)));
    CHECK(cudaMemcpy(upd, &_upd, sizeof(int), cudaMemcpyHostToDevice));
    //call_solve<<<1, 32>>>(ppr, s, t, flow, upd);
    void *param[] = {&ppr, &s, &t, &flow, &upd};
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
    cudaSetDevice(2);

    int n, m, s, t;
    auto input = read_max_file(n, m, s, t);
    debug("read done");
    PreflowPushRelabel flow(n, m, input);
    //debug("test", n, m, s, t);
    cout << solve(flow, s, t) << endl;


}
