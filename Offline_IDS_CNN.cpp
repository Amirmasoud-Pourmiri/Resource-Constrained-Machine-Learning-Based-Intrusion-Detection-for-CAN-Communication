// ids_labeled_pycompat.cpp
// 10-frame sliding windows with labels, sklearn-compatible MinMax scaling,
// DOT-based decision tree inference, and accuracy computation.
// Parsing now column-aware: hex for ID/D0-D7, decimal for Timestamp/DLC.
//
// Build: g++ -O2 -std=c++17 ids_labeled_pycompat.cpp -o ids
// Run:   ./ids
//
// Needs: pruned_DT_explanation.dot  (your exported/pruned tree)
//        train.csv (the data lines, one per line)

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <limits>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

/* ───────── Config ───────── */
static constexpr int  SEQ_LEN  = 10;
static const char*    DOT_FILE = "pruned_tree.dot";
static constexpr double PROBA_THRESH = 0.50;   // raise (e.g., 0.65) to reduce false positives
static const char*    DATA_FILE = "train.csv";
static constexpr int  PER_FRAME = 11;

/* ───────── Your pasted MinMax vectors (from Python) ─────────
   Use the new vectors emitted by the second Python code. */
std::vector<float> min_vals = {
    0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f,
    0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f,
    0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f,
    0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f,
    0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f,
    0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f,
    0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f,
    0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f,
    0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f,
    0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f
};
    // ... (paste the new min_vals here)

std::vector<float> max_vals = {
    1481193344.00000000f, 1680.00000000f, 8.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f,
    1481193344.00000000f, 1680.00000000f, 8.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f,
    1481193344.00000000f, 1680.00000000f, 8.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f,
    1481193344.00000000f, 1680.00000000f, 8.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f,
    1481193344.00000000f, 1680.00000000f, 8.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f,
    1481193344.00000000f, 1680.00000000f, 8.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f,
    1481193344.00000000f, 1680.00000000f, 8.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f,
    1481193344.00000000f, 1680.00000000f, 8.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f,
    1481193344.00000000f, 1680.00000000f, 8.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f,
    1481193344.00000000f, 1680.00000000f, 8.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f, 255.00000000f
};


/* ───────── Types ───────── */
struct Node { int feat=-1; float thr=0.f; int left=-1; int right=-1; int val0=0; int val1=0; };
using Tree = std::vector<Node>;
struct LabeledFrame { std::vector<float> feat; int label=-1; };
struct Win { std::vector<float> x; int y=-1; };

/* ───────── Small utils ───────── */
static inline std::string ltrim(const std::string& s){ size_t i=0; while(i<s.size() && std::isspace((unsigned char)s[i])) ++i; return s.substr(i); }
static inline std::string rtrim(const std::string& s){ size_t i=s.size(); while(i>0 && std::isspace((unsigned char)s[i-1])) --i; return s.substr(0,i); }
static inline std::string trim (const std::string& s){ return rtrim(ltrim(s)); }

static std::vector<std::string> split_csv(const std::string& line){
    std::vector<std::string> out; std::stringstream ss(line); std::string tok;
    while (std::getline(ss, tok, ',')) {
        tok = trim(tok);
        out.push_back(tok);  // Push even if empty
    }
    return out;
}

/* ───────── Column-aware parsing (matches updated Python) ─────────
   - Hex for ID/D0-D7: strip optional "0x", parse as hex, bad => 0
   - Decimal for Timestamp/DLC/others: parse as float, bad => 0     */
static float parse_hex_py_compat(const std::string& s) {
    std::string t = trim(s);
    if (t.empty()) return 0.0f;
    const char* p = t.c_str();
    if (t.size() > 2 && t[0] == '0' && (t[1] == 'x' || t[1] == 'X')) p += 2;
    char* endp = nullptr;
    long v = std::strtol(p, &endp, 16);
    if (endp && *endp == '\0') return static_cast<float>(v);
    return 0.0f;
}

static float parse_dec_py_compat(const std::string& s) {
    std::string t = trim(s);
    if (t.empty()) return 0.0f;
    char* endp = nullptr;
    float fv = std::strtof(t.c_str(), &endp);
    if (endp && *endp == '\0') return fv;
    return 0.0f;
}

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

/* ───────── Build windows like Python (label = last frame) ───────── */
std::vector<Win> make_windows_with_labels(const std::vector<LabeledFrame>& frames,
                                          int seq_len, int per_frame)
{
    std::vector<Win> out;
    if ((int)frames.size() < seq_len) return out;
    const int D = seq_len * per_frame;
    for (int i=0; i <= (int)frames.size()-seq_len; ++i) {
        std::vector<float> flat(D);
        for (int t=0; t<seq_len; ++t) {
            const auto& fr = frames[i+t].feat;
            std::copy(fr.begin(), fr.end(), flat.begin() + t*per_frame);
        }
        int y = frames[i+seq_len-1].label;
        out.push_back({std::move(flat), y});
    }
    return out;
}

/* ───────── Scale windows using sklearn MinMax (no clip) ───────── */
void minmax_scale_inplace(std::vector<Win>& windows,
                          const std::vector<float>& mins,
                          const std::vector<float>& maxs)
{
    if (windows.empty()) return;
    const size_t D = mins.size();
    if (maxs.size() != D) throw std::runtime_error("min/max size mismatch");
    for (auto& w : windows) {
        if (w.x.size() != D) throw std::runtime_error("window len != min/max len");
        for (size_t j=0;j<D;++j) {
            float denom = maxs[j] - mins[j];
            w.x[j] = (denom!=0.f) ? (w.x[j] - mins[j]) / denom : 0.f;
        }
    }
}

/* ───────── Load your labeled frames (skip incomplete) ─────────
   Expect 12 tokens: TS,ID,DLC,D0..D7,label
   Lines with <12 fields skipped. Label != "1" => 0.     */
std::vector<LabeledFrame> load_frames_from_file(const std::string& path){
    std::ifstream in(path);
    if(!in) throw std::runtime_error("Data file not found: " + path);

    std::string line;
    std::vector<LabeledFrame> frames;
    int lineno = 0;
    while (std::getline(in, line)) {
        ++lineno; line = trim(line);
        if (line.empty()) continue;
        auto toks = split_csv(line);
        if (toks.empty()) continue;

        if ((int)toks.size() != 12) {
            std::cerr << "[Skip] line " << lineno << ": " << toks.size()
                      << " fields; expected 12 (TS,ID,DLC,D0..D7,Label)\n";
            continue;
        }

        int label = (toks.back()=="1") ? 1 : 0;
        toks.pop_back();

        std::vector<float> fr; fr.reserve(11);
        fr.push_back(parse_dec_py_compat(toks[0])); // Timestamp: dec
        fr.push_back(parse_hex_py_compat(toks[1])); // ID: hex
        fr.push_back(parse_dec_py_compat(toks[2])); // DLC: dec
        for (int k=3; k<11; ++k)                    // D0..D7: hex
            fr.push_back(parse_hex_py_compat(toks[k]));

        frames.push_back({std::move(fr), label});
    }
    return frames;
}

/* ───────── main ───────── */
int main(){
    try{
        // 1) Parse DOT
        int root=0, maxf_rep=-1;
        Tree tree = parse_dot(DOT_FILE, root, maxf_rep);
        int expected_len = SEQ_LEN * PER_FRAME;
        std::cout << "DOT parsed. Per-frame="<<PER_FRAME<<" | expected window len="<<expected_len<<"\n";

        // 2) Prepare MinMax vectors to match expected length
        auto mins = prepare_minmax(min_vals, PER_FRAME, expected_len, "min_vals");
        auto maxs = prepare_minmax(max_vals, PER_FRAME, expected_len, "max_vals");

        // 3) Load labeled frames (skipping incomplete rows)
        auto frames = load_frames_from_file(DATA_FILE);
        std::cout << "Frames kept: " << frames.size() << "\n";
        if ((int)frames.size() < SEQ_LEN) { std::cerr << "Not enough frames.\n"; return 0; }

        // 4) Build windows (label = label of last frame in window)
        auto windows = make_windows_with_labels(frames, SEQ_LEN, PER_FRAME);
        if (windows.empty()) { std::cerr << "No windows built.\n"; return 0; }
        std::cout << "Windows built: " << windows.size() << "\n";

        // 5) Scale like sklearn MinMax (no clipping)
        minmax_scale_inplace(windows, mins, maxs);
        std::cerr << "[Scaling] Applied pasted MinMax to flattened windows\n";

        // 6) Predict each window & print (optional; comment out for large data)
        // for (size_t k=0;k<windows.size();++k) {
        //     auto pr = predict_proba(tree, root, windows[k].x);
        //     std::cout << "Win " << k << " => " << (pr.hard? "class1":"class0")
        //               << "  (p1=" << pr.p1 << ")  | label=" << windows[k].y << "\n";
        // }

        // 7) Accuracy in two interpretations (since we don't know which class is "malicious")
        int TPm=0,TNm=0,FPm=0,FNm=0, Nm=0; // assuming class1 = malicious
        int TPn=0,TNn=0,FPn=0,FNn=0, Nn=0; // assuming class1 = normal

        for (const auto& w: windows) {
            if (w.y!=0 && w.y!=1) continue;
            auto pr = predict_proba(tree, root, w.x);
            // mapping A: class1 == malicious
            {
                int pred_mal = pr.hard;   // 1 => malicious
                int y_mal    = w.y;       // assumes label '1' means malicious
                ++Nm;
                if (pred_mal==1 && y_mal==1) ++TPm;
                else if (pred_mal==0 && y_mal==0) ++TNm;
                else if (pred_mal==1 && y_mal==0) ++FPm;
                else ++FNm;
            }
            // mapping B: class1 == normal
            {
                int pred_norm = pr.hard;     // 1 => class1
                int y_norm    = w.y;         // assumes label '1' means normal
                ++Nn;
                if (pred_norm==1 && y_norm==1) ++TPn;
                else if (pred_norm==0 && y_norm==0) ++TNn;
                else if (pred_norm==1 && y_norm==0) ++FPn;
                else ++FNn;
            }
        }

        auto safe_div = [](int a, int b)->double{ return b? double(a)/b : 0.0; };

        std::cout << "\n--- Metrics assuming label '1' = MALICIOUS ---\n";
        std::cout << "Evaluated windows: " << Nm << "\n";
        std::cout << "TP="<<TPm<<" TN="<<TNm<<" FP="<<FPm<<" FN="<<FNm<<"\n";
        std::cout << "Accuracy="<<safe_div(TPm+TNm,Nm)
                  << "  Precision="<<safe_div(TPm,TPm+FPm)
                  << "  Recall="<<safe_div(TPm,TPm+FNm)
                  << "  F1="<<safe_div(2*TPm,2*TPm+FPm+FNm) << "\n";

        std::cout << "\n--- Metrics assuming label '1' = NORMAL ---\n";
        std::cout << "Evaluated windows: " << Nn << "\n";
        std::cout << "TP="<<TPn<<" TN="<<TNn<<" FP="<<FPn<<" FN="<<FNn<<"\n";
        std::cout << "Accuracy="<<safe_div(TPn+TNn,Nn)
                  << "  Precision="<<safe_div(TPn,TPn+FPn)
                  << "  Recall="<<safe_div(TPn,TPn+FNn)
                  << "  F1="<<safe_div(2*TPn,2*TPn+FPn+FNn) << "\n";
    }
    catch(const std::exception& e){
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
