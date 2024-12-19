#include <bits/stdc++.h>
using namespace std;

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
struct FlowNetwork {
    int n;
    vector<vector<edge<T>>> g;
    vector<T> excess;
    FlowNetwork(int _n): n(_n), g(n), excess(n) {}
    void add_edge(int u, int v, T cap) {
        g[u].pb(edge<T>{u, v, SZ(g[v]), cap, T(0)});
        g[v].pb(edge<T>{v, u, SZ(g[u]) - 1, T(0), T(0)});
    }
    void add_flow(edge<T> &e, T f) {
        e.flow += f;
        g[e.to][e.rev].flow -= f;
        excess[e.from] -= f;
        excess[e.to] += f;
    }
};

template <class T>
struct PreflowPushRelabel : FlowNetwork<T> {
    vector<int> h;
    PreflowPushRelabel(int _n): FlowNetwork<T>(_n), h(_n) {}
    bool push(edge<T> &e) {
        if (FlowNetwork<T>::excess[e.from] <= 0 || e.cap - e.flow <= 0 || h[e.from] != h[e.to] + 1)
            return false;
        T f = min(FlowNetwork<T>::excess[e.from], e.cap - e.flow);
        FlowNetwork<T>::add_flow(e, f);
        return true;
    }
    bool relabel(int u) {
        if (FlowNetwork<T>::excess[u] <= 0) return false;
        bool has_edge = false;
        for (auto &e : FlowNetwork<T>::g[u]) {
            if (e.flow < e.cap) {
                has_edge = true;
                if (h[u] > h[e.to])
                    return false;
            }
        }
        if (!has_edge) return false;
        int min_h = numeric_limits<int>::max();
        for (auto &e : FlowNetwork<T>::g[u])
            if (e.flow < e.cap)
                min_h = min(min_h, h[e.to]);
        h[u] = min_h + 1;
        return true;
    }
    void init(int s) {
        h[s] = FlowNetwork<T>::n;
        for (auto &e : FlowNetwork<T>::g[s]) {
            if (e.cap == 0) continue;
            FlowNetwork<T>::add_flow(e, e.cap);
        }
    }
    T solve(int s, int t) {
        init(s);
        int k = 0;
        while (true) {
            k++;
            bool ok = false;
            for (int i = 0; i < FlowNetwork<T>::n; i++) {
                if (i == t || FlowNetwork<T>::excess[i] <= 0) continue;
                if (i != t && relabel(i)) ok = true;
                for (auto &e : FlowNetwork<T>::g[i])
                    if (push(e)) ok = true;
            }
            if (k % 100 == 0) debug("ok", k, FlowNetwork<T>::excess[t]);
            if (!ok) break;
        }
        debug("k", k);
        return FlowNetwork<T>::excess[t];
    }
};

PreflowPushRelabel<ll> read_input(int &n, int &m, int &s, int &t) {
    cin >> n >> m >> s >> t;
    s--; t--;
    PreflowPushRelabel<ll> flow(n);
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
    PreflowPushRelabel<ll> flow(n);
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
            int u, v, cap;
            ss >> u >> v >> cap;
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
    //auto flow = read_input(n, m, s, t);
    auto flow = read_max_file(n, m, s, t);
    debug("test", n, m, s, t);
    cout << flow.solve(s, t) << endl;


}
