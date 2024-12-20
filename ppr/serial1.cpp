#include <bits/stdc++.h>

using namespace std;

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
struct PreflowPushRelabel {
    int n, m;
    vector<vector<edge<T>>> g;
    vector<edge<T>> pool;
    vector<T> excess;
    vector<int> h;
    PreflowPushRelabel(int _n, int _m): n(_n), m(_m), g(n), pool(n), excess(n), h(n) {}
    void add_edge(int u, int v, T cap) {
        g[u].pb(edge<T>({u, v, SZ(g[v]), cap, T(0)}));
        g[v].pb(edge<T>({v, u, SZ(g[u]) - 1, T(0), T(0)}));
    }
    void add_flow(edge<T> &e, T f) {
        e.flow += f;
        g[e.to][e.rev].flow -= f;
        excess[e.from] += -f;
        excess[e.to] += f;
    }
    bool push(edge<T> &e) {
        if (excess[e.from] <= 0 || e.cap - e.flow <= 0 || h[e.from] != h[e.to] + 1)
            return false;
        T f = min(excess[e.from], e.cap - e.flow);
        add_flow(e, f);
        return true;
    }
    bool relabel(int u) {
        if (excess[u] <= 0) return false;
        int min_h = INT_MAX;
        for (auto &e : g[u]) {
            if (e.flow < e.cap) {
                min_h = min(min_h, h[e.to]);
            }
        }
        if (min_h == INT_MAX || min_h < h[u]) return false;
        h[u] = min_h + 1;
        return true;
    }
    T solve(int s, int t) {
        h[s] = n;
        for (auto &e : g[s]) {
            add_flow(e, e.cap);
        }
        for (int k = 0; ; k++) {
            bool ok = false;
            for (int i = 0; i < n; i++) {
                if (i == t || excess[i] <= 0) continue;
                for (auto &e : g[i]) {
                    if (push(e)) ok = true;
                }
            }
            for (int i = 0; i < n; i++) {
                if (i == t || excess[i] <= 0) continue;
                if (i != t && relabel(i)) ok = true;
            }
            if (!ok) break;
        }
        return excess[t];
    }
};

PreflowPushRelabel<ll> read_input(int &n, int &m, int &s, int &t) {
    cin >> n >> m >> s >> t;
    s--; t--;
    PreflowPushRelabel<ll> flow(n, m);
    for (int i = 0; i < m; i++) {
        int u, v, cap;
        cin >> u >> v >> cap;
        u--; v--;
        flow.add_edge(u, v, cap);
    }
    return flow;
}

PreflowPushRelabel<ll> read_max_file(int &n, int &m, int &s, int &t) {
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
    PreflowPushRelabel<ll> flow(n, m);
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
            //if (cnt % 10000 == 0) debug("read", cnt);
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

    int n, m, s, t;

    auto read_start = std::chrono::high_resolution_clock::now();
    //auto flow = read_max_file(n, m, s, t);
    auto flow = read_input(n, m, s, t);
    auto read_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> read_time = read_end - read_start;
    std::cout << "Read Time: " << read_time.count() * 1000.0 << " ms" << std::endl;

    auto solve_start = std::chrono::high_resolution_clock::now();
    cout << flow.solve(s, t) << endl;
    auto solve_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = solve_end - solve_start;
    std::cout << "Solve Time: " << solve_time.count() * 1000.0 << " ms" << std::endl;

    // max flow check
    for (int i = 0; i < n; i++) {
        if (i == s || i == t) continue;
        ll total = 0;
        for (auto &e : flow.g[i])
            total += e.flow;
        assert(total == 0);
    }
    for (int i = 0; i < n; i++)
        for (auto &e : flow.g[i])
            assert(e.flow <= e.cap);
    queue<int> q;
    q.push(s);
    vector<bool> vst(n);
    vst[s] = true;
    while (!q.empty()) {
        int now = q.front();
        q.pop();
        for (auto &e : flow.g[now]) {
            if (e.cap == e.flow) continue;
            if (vst[e.to]) continue;
            vst[e.to] = true;
            q.push(e.to);
        }
    }
    assert(!vst[t]);

}
