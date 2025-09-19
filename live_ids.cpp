// ids_live.cpp
// Live CAN reader (socketCAN) -> 10-frame sliding windows -> MinMax scaling
// -> DOT decision tree inference. No labels required.
//
// Build: g++ -O2 -std=c++17 ids_live.cpp -o ids
// Run:   sudo ./ids --live can0
//
// Needs: pruned_DT_explanation.dot (exported/pruned tree)

#include <algorithm>
#include <cctype>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

/* SocketCAN */
#include <linux/can.h>
#include <linux/can/raw.h>
#include <net/if.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

/* ───────── Config ───────── */
static constexpr int  SEQ_LEN  = 10;          // 10 frames per window
static const char*    DOT_FILE = "pruned_DT_explanation.dot";
static constexpr double PROBA_THRESH = 0.50;  // raise (e.g. 0.65) to reduce false positives

/* ───────── Min/Max (per frame = 11 values) ─────────
   Order per frame must be:
   ['Timestamp','ID','DLC','D0','D1','D2','D3','D4','D5','D6','D7']  */

std::vector<float> min_vals = {
    /* Put EXACTLY 11 numbers here (one frame). Example: */
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
};

std::vector<float> max_vals = {
    /* Put EXACTLY 11 numbers here (one frame). Example using your scales: */
    1481193344.0f, 50000.0f, 8.0f, 99.0f, 99.0f, 99.0f, 99.0f, 99.0f, 99.0f, 99.0f, 99.0f
};

/* ───────── Types ───────── */
struct Node { int feat=-1; float thr=0.f; int left=-1; int right=-1; int val0=0; int val1=0; };
using Tree = std::vector<Node>;
struct Win { std::vector<float> x; };  // no label in live mode

/* ───────── Trims (used by DOT parser) ───────── */
static inline std::string ltrim(const std::string& s){ size_t i=0; while(i<s.size() && std::isspace((unsigned char)s[i])) ++i; return s.substr(i); }
static inline std::string rtrim(const std::string& s){ size_t i=s.size(); while(i>0 && std::isspace((unsigned char)s[i-1])) --i; return s.substr(0,i); }
static inline std::string trim (const std::string& s){ return rtrim(ltrim(s)); }

/* ───────── DOT parser ───────── */
Tree parse_dot(const std::string& path, int& root_idx, int& max_feat_idx){
    std::ifstream in(path);
    if(!in) throw std::runtime_error("DOT file not found: " + path);

    std::regex rg_node(R"(^\s*(\d+)\s+\[label="x(\d+)\s+<=\s+([0-9eE+.\-]+).*value\s*=\s*\[(\d+),\s*(\d+)])");
    std::regex rg_leaf(R"(^\s*(\d+)\s+\[label="gini.*value\s*=\s*\[(\d+),\s*(\d+)])");
    std::regex rg_edge(R"(^\s*(\d+)\s*->\s*(\d+))");

    std::unordered_map<int,int> id2idx, child_seen;
    struct Edge { int p,c; };
    std::vector<Edge> edges;
    Tree tree; max_feat_idx = -1;

    std::string line; std::smatch m;
    while (std::getline(in, line)) {
        if (std::regex_search(line, m, rg_node)) {
            int id = std::stoi(m[1]); Node n;
            n.feat = std::stoi(m[2]); n.thr = std::stof(m[3]);
            n.val0 = std::stoi(m[4]); n.val1 = std::stoi(m[5]);
            id2idx[id] = (int)tree.size(); tree.push_back(n);
            max_feat_idx = std::max(max_feat_idx, n.feat);
        } else if (std::regex_search(line, m, rg_leaf)) {
            int id = std::stoi(m[1]); Node n; n.val0 = std::stoi(m[2]); n.val1 = std::stoi(m[3]);
            id2idx[id] = (int)tree.size(); tree.push_back(n);
        } else if (std::regex_search(line, m, rg_edge)) {
            edges.push_back({ std::stoi(m[1]), std::stoi(m[2]) });
            child_seen[edges.back().c]++;
        }
    }
    for (auto& e: edges) {
        if (!id2idx.count(e.p) || !id2idx.count(e.c)) continue;
        Node& p = tree[id2idx[e.p]];
        if (p.left==-1) p.left=id2idx[e.c]; else p.right=id2idx[e.c];
    }
    for (auto& kv : id2idx) {
        if (!child_seen.count(kv.first)) { root_idx = kv.second; return tree; }
    }
    throw std::runtime_error("Root node not found");
}

/* ───────── Prediction (with leaf prob) ───────── */
static int descend_leaf(const Node& n, const Tree& t, const std::vector<float>& x){
    if (n.left==-1 && n.right==-1) return int(&n - &t[0]);
    if (n.feat < 0 || n.feat >= (int)x.size()) throw std::runtime_error("Invalid feat x"+std::to_string(n.feat));
    int child = (x[n.feat] <= n.thr) ? n.left : n.right;
    if (child < 0 || child >= (int)t.size()) throw std::runtime_error("Invalid child idx");
    return descend_leaf(t[child], t, x);
}
struct Pred { int hard; double p1; };
Pred predict_proba(const Tree& t, int root, const std::vector<float>& x){
    int leaf = descend_leaf(t[root], t, x);
    const Node& L = t[leaf];
    double n0 = std::max(0, L.val0), n1 = std::max(0, L.val1);
    double p1 = (n0+n1>0) ? n1/(n0+n1) : 0.5;
    int hard = (p1 >= PROBA_THRESH) ? 1 : 0;
    return {hard, p1};
}

/* ───────── Helpers ───────── */
std::vector<float> tile_per_frame(const std::vector<float>& base, int times) {
    std::vector<float> out; out.reserve(base.size()*times);
    for (int i=0;i<times;++i) out.insert(out.end(), base.begin(), base.end());
    return out;
}
std::vector<float> prepare_minmax(const std::vector<float>& v, int per_frame, int expected_len, const char* name){
    if ((int)v.size()==expected_len) return v;
    if ((int)v.size()==per_frame)    return tile_per_frame(v, expected_len/per_frame);
    if ((int)v.size()> per_frame) {
        std::cerr << "[WARN] " << name << " size " << v.size()
                  << " != " << expected_len << "; using first " << per_frame << " and tiling.\n";
        std::vector<float> base(v.begin(), v.begin()+per_frame);
        return tile_per_frame(base, expected_len/per_frame);
    }
    throw std::runtime_error(std::string(name) + " too short");
}

/* ───────── SocketCAN: open + read ───────── */
int open_can_socket(const std::string& ifname){
    int s = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (s < 0) throw std::runtime_error("socket(PF_CAN) failed");

    // bind to interface
    struct ifreq ifr{};
    std::snprintf(ifr.ifr_name, sizeof(ifr.ifr_name), "%s", ifname.c_str());
    if (ioctl(s, SIOCGIFINDEX, &ifr) < 0) {
        close(s);
        throw std::runtime_error("Unknown CAN interface: " + ifname);
    }
    sockaddr_can addr{};
    addr.can_family  = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;
    if (bind(s, (sockaddr*)&addr, sizeof(addr)) < 0) {
        close(s);
        throw std::runtime_error("bind(can) failed");
    }

    // non-blocking poll based read
    int on = 1;
    setsockopt(s, SOL_CAN_RAW, CAN_RAW_RECV_OWN_MSGS, &on, sizeof(on)); // optional
    return s;
}

static inline double now_seconds(){
    timespec ts{};
    clock_gettime(CLOCK_BOOTTIME, &ts); // monotonic since boot
    return ts.tv_sec + ts.tv_nsec*1e-9;
}

/* ───────── Frame -> 11 features ───────── */
static inline void frame_to_features(const can_frame& f, double t0, std::array<float,11>& feat){
    // Timestamp as "seconds since start", similar scale to your training if min/max match
    double t_rel = now_seconds() - t0;

    uint32_t id = f.can_id & (CAN_EFF_FLAG ? CAN_EFF_MASK : CAN_SFF_MASK); // numeric ID
    feat[0]  = (float)t_rel;                 // Timestamp
    feat[1]  = (float)id;                    // ID (decimal)
    feat[2]  = (float)f.can_dlc;             // DLC
    // D0..D7 (pad with zeros if DLC<8)
    for (int i=0;i<8;++i) feat[3+i] = (i < f.can_dlc) ? (float)f.data[i] : 0.0f;
}

/* ───────── Build + scale a window from a deque of per-frame features ───────── */
void build_window_flattened(const std::deque<std::array<float,11>>& buf, std::vector<float>& flat){
    const int per_frame = 11;
    flat.resize(SEQ_LEN * per_frame);
    for (int t=0;t<SEQ_LEN;++t){
        const auto& fr = buf[t];
        std::copy(fr.begin(), fr.end(), flat.begin() + t*per_frame);
    }
}

void minmax_scale_inplace( std::vector<float>& x,
                           const std::vector<float>& mins,
                           const std::vector<float>& maxs )
{
    if (x.size()!=mins.size() || maxs.size()!=mins.size())
        throw std::runtime_error("min/max length mismatch");
    for (size_t j=0;j<x.size();++j){
        float denom = maxs[j]-mins[j];
        x[j] = (denom!=0.f) ? (x[j]-mins[j])/denom : 0.f;
    }
}

/* ───────── main ───────── */
int main(int argc, char** argv){
    try{
        // CLI: --live <iface>
        std::string live_iface;
        for (int i=1;i<argc;++i){
            std::string a = argv[i];
            if (a=="--live" && i+1<argc) { live_iface = argv[++i]; }
        }
        if (live_iface.empty()){
            std::cerr << "Usage: sudo ./ids --live <can-iface>\n"
                         "Example: sudo ./ids --live can0\n";
            return 1;
        }

        // 1) Parse DOT & infer per-frame width (>=100 index => 11 per frame)
        int root=0, maxf_rep=-1;
        Tree tree = parse_dot(DOT_FILE, root, maxf_rep);
        bool any_100=false; int maxf=-1;
        for (const auto& n: tree) if (n.feat>=0) { if (n.feat>=100) any_100=true; maxf=std::max(maxf,n.feat); }
        int per_frame = any_100 ? 11 : 10;
        int expected_len = SEQ_LEN * per_frame;
        std::cout << "DOT parsed. Per-frame="<<per_frame<<" | expected window len="<<expected_len<<"\n";
        if (per_frame!=11) {
            std::cerr << "ERROR: This model expects "<<per_frame<<" features per frame. Live reader emits 11.\n";
            return 1;
        }

        // 2) Prepare MinMax vectors to match expected length
        auto mins = prepare_minmax(min_vals, per_frame, expected_len, "min_vals");
        auto maxs = prepare_minmax(max_vals, per_frame, expected_len, "max_vals");

        // 3) Open CAN and start reading
        int s = open_can_socket(live_iface);
        std::cout << "Listening on " << live_iface << " … (Ctrl+C to stop)\n";

        std::deque<std::array<float,11>> ring;   // last SEQ_LEN frames of features
        std::vector<float> flat;                 // flattened + scaled window
        double t0 = now_seconds();

        pollfd pfd{ s, POLLIN, 0 };

        while (true){
            int r = poll(&pfd, 1, /*ms*/ 1000);
            if (r < 0) { perror("poll"); break; }
            if (r == 0) { continue; }

            if (pfd.revents & POLLIN){
                can_frame f{};
                ssize_t n = read(s, &f, sizeof(f));
                if (n != (ssize_t)sizeof(f)) continue;

                // build 11-dim feature for this frame
                std::array<float,11> feat{};
                frame_to_features(f, t0, feat);

                // push to ring buffer
                ring.push_back(feat);
                if ((int)ring.size() > SEQ_LEN) ring.pop_front();

                // when we have 10 frames, build a window and classify
                if ((int)ring.size() == SEQ_LEN){
                    build_window_flattened(ring, flat);
                    // scale with sklearn-compatible MinMax
                    minmax_scale_inplace(flat, mins, maxs);
                    // predict
                    Pred pr = predict_proba(tree, root, flat);

                    // Print like: ID=440 DLC=8 p1=0.84 -> class1
                    uint32_t id = f.can_id & (f.can_id & CAN_EFF_FLAG ? CAN_EFF_MASK : CAN_SFF_MASK);
                    std::cout << "ID=" << id
                              << " DLC=" << (int)f.can_dlc
                              << "  p1=" << pr.p1
                              << "  => " << (pr.hard ? "class1" : "class0")
                              << std::endl;
                }
            }
        }

        close(s);
    }
    catch(const std::exception& e){
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
