// GD AI Learner - adds an "AI Menu" to the pause screen. A Q-learning agent takes
// over the level you are on, dying, respawning and improving every attempt.

#include <Geode/Geode.hpp>
#include <Geode/modify/PauseLayer.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/GJBaseGameLayer.hpp>
#include <unordered_map>
#include <array>
#include <vector>
#include <fstream>
#include <random>
#include <algorithm>
#include <chrono>

using namespace geode::prelude;

// ---------- the learner ----------

struct Step {
    uint64_t key;    // position-based state (memorises this level)
    uint64_t gkey;   // feature-based state (what's ahead; carries over to other levels)
    int action;
    float reward;
};

struct Obstacle {
    float x;
    float y;
    bool hazard;
};

struct Agent {
    bool active = false;
    bool episodeOver = false;
    int levelID = 0;
    float levelLength = 0.f;
    int attempts = 0;
    float bestX = 0.f;
    float startX = 0.f;
    float lastX = 0.f;
    bool holding = false;
    int tick = 0;

    // "Learn Rapidly": faster game speed, instant respawn, bigger learning steps.
    bool rapid = false;
    bool useCheckpoints = true;   // false = rapid learning in normal mode from 0%
    int deadTicks = 0;
    bool skipDelayedReset = false;

    // Rapid mode places practice checkpoints just behind the frontier so each
    // attempt replays only the part it hasn't mastered yet.
    std::vector<float> checkpointXs;
    int deathsSinceProgress = 0;
    float bestAtLastCheck = 0.f;

    // After beating the level in practice, try a real run in normal mode.
    // A death there sends it back to practice to train that part again.
    bool verifying = false;
    enum class Switch { None, ToNormal, ToPractice } pendingSwitch = Switch::None;

    // Input timelines keyed by X position (best run, player guide).
    struct Input {
        float x;
        bool hold;
    };
    std::vector<Input> attemptInputs;
    std::vector<Input> bestRun;
    float bestRunEnd = 0.f;
    bool replaying = false;
    size_t replayIdx = 0;

    // Guide: the player records the start of the level; the AI copies it there.
    std::vector<Input> guide;
    float guideEnd = 0.f;
    bool recording = false;
    bool useGuide = true;

    // Auto-tuned exploration: the longer it's stuck at the frontier, the more it explores.
    int frontierDeaths = 0;
    float frontierBest = 0.f;

    // Stats for the graph / heatmap.
    std::vector<float> history;                   // % reached per attempt
    std::vector<double> attemptTimes;             // seconds since start, for attempts/min
    float trainedSeconds = 0.f;
    std::unordered_map<uint32_t, int> deathMap;   // (x/30, y/30) -> deaths
    bool showHeatmap = false;

    std::unordered_map<uint64_t, std::array<float, 2>> q;
    std::unordered_map<uint64_t, std::array<float, 2>> gq;   // shared across all levels
    bool gqLoaded = false;
    std::vector<Obstacle> obstacles;
    std::vector<Step> trajectory;
    std::mt19937 rng{std::random_device{}()};

    static constexpr int DECIDE_EVERY = 3;   // physics ticks between decisions
    float alpha() const { return rapid ? 0.5f : 0.3f; }
    static constexpr float GAMMA = 0.98f;

    static double now() {
        using namespace std::chrono;
        return duration<double>(steady_clock::now().time_since_epoch()).count();
    }

    float pctOf(float x) const {
        return levelLength > 0 ? std::min(100.f, 100.f * x / levelLength) : 0.f;
    }

    // The best run is a single real attempt from the start of the level, so a
    // replay is an exact copy (no stitched practice segments).
    void commitAttempt(float endX) {
        if (startX > 50.f || endX <= bestRunEnd) return;
        bestRun = attemptInputs;
        bestRunEnd = endX;
    }

    // Called once per attempt that ends (death or finish).
    void recordAttemptEnd(float endX, float endY, bool died) {
        history.push_back(pctOf(endX));
        if (history.size() > 400) history.erase(history.begin());
        attemptTimes.push_back(now());
        if (attemptTimes.size() > 200) attemptTimes.erase(attemptTimes.begin());
        if (died) {
            int xb = static_cast<int>(endX / 30.f);
            int yb = std::clamp(static_cast<int>(endY / 30.f), 0, 255);
            deathMap[(static_cast<uint32_t>(xb) << 8) | static_cast<uint32_t>(yb)]++;
            if (bestX > frontierBest + 30.f) {
                frontierBest = bestX;
                frontierDeaths = 0;
            } else {
                frontierDeaths++;
            }
        }
    }

    float attemptsPerMinute() const {
        if (attemptTimes.size() < 2) return 0.f;
        double span = attemptTimes.back() - attemptTimes.front();
        return span > 0 ? static_cast<float>((attemptTimes.size() - 1) * 60.0 / span) : 0.f;
    }

    bool guideHoldAt(float x) const {
        auto it = std::upper_bound(guide.begin(), guide.end(), x,
            [](float v, Input const& in) { return v < in.x; });
        return it == guide.begin() ? false : std::prev(it)->hold;
    }

    static int modeOf(PlayerObject* p) {
        return p->m_isShip ? 1 : p->m_isBall ? 2 : p->m_isBird ? 3 : p->m_isDart ? 4
             : p->m_isRobot ? 5 : p->m_isSpider ? 6 : p->m_isSwing ? 7 : 0;
    }

    // State = where we are in the level + vertical position/velocity + gamemode.
    // Levels are deterministic, so position-based states let the agent learn
    // exactly where to click on this level.
    static uint64_t makeKey(PlayerObject* p, bool holding) {
        int xb = static_cast<int>(p->getPositionX() / 10.f);
        int yb = static_cast<int>(p->getPositionY() / 15.f);
        double vy = p->m_yVelocity;
        int vb = vy > 2 ? 2 : (vy < -2 ? 0 : 1);
        uint64_t k = static_cast<uint32_t>(xb);
        k = (k << 12) | (static_cast<uint32_t>(yb + 100) & 0xFFF);
        k = (k << 2) | vb;
        k = (k << 3) | modeOf(p);
        k = (k << 1) | (p->m_isUpsideDown ? 1 : 0);
        k = (k << 1) | (holding ? 1 : 0);
        return k;
    }

    void buildObstacles(GJBaseGameLayer* gl) {
        obstacles.clear();
        if (!gl->m_objects) return;
        for (auto obj : CCArrayExt<GameObject*>(gl->m_objects)) {
            if (!obj) continue;
            bool hazard = obj->m_objectType == GameObjectType::Hazard;
            bool solid = obj->m_objectType == GameObjectType::Solid;
            if (!hazard && !solid) continue;
            obstacles.push_back({obj->getPositionX(), obj->getPositionY(), hazard});
        }
        std::sort(obstacles.begin(), obstacles.end(),
            [](Obstacle const& a, Obstacle const& b) { return a.x < b.x; });
    }

    // State = what's coming up (nearest spike and block ahead, relative to the
    // player) + vertical velocity + gamemode. Not tied to this level.
    uint64_t makeFeatureKey(PlayerObject* p, bool holding) const {
        float px = p->getPositionX();
        float py = p->getPositionY();
        int hdx = 31, hdy = 0, sdx = 31, sdy = 0;
        bool gotH = false, gotS = false;
        auto it = std::lower_bound(obstacles.begin(), obstacles.end(), px - 30.f,
            [](Obstacle const& o, float v) { return o.x < v; });
        for (; it != obstacles.end() && it->x < px + 360.f && !(gotH && gotS); ++it) {
            float dy = it->y - py;
            if (std::abs(dy) > 150.f) continue;
            int dxb = std::clamp(static_cast<int>((it->x - px) / 15.f) + 2, 0, 30);
            int dyb = std::clamp(static_cast<int>(std::floor(dy / 20.f)), -7, 7);
            if (it->hazard && !gotH) {
                hdx = dxb;
                hdy = dyb;
                gotH = true;
            } else if (!it->hazard && !gotS) {
                sdx = dxb;
                sdy = dyb;
                gotS = true;
            }
        }
        double vy = p->m_yVelocity;
        int vb = vy > 2 ? 2 : (vy < -2 ? 0 : 1);
        uint64_t k = static_cast<uint32_t>(hdx);
        k = (k << 4) | static_cast<uint32_t>(hdy + 7);
        k = (k << 5) | static_cast<uint32_t>(sdx);
        k = (k << 4) | static_cast<uint32_t>(sdy + 7);
        k = (k << 2) | vb;
        k = (k << 3) | modeOf(p);
        k = (k << 1) | (p->m_isUpsideDown ? 1 : 0);
        k = (k << 1) | (p->m_isOnGround ? 1 : 0);
        k = (k << 1) | (holding ? 1 : 0);
        return k;
    }

    int choose(uint64_t key, uint64_t gkey, float x) {
        bool smart = Mod::get()->getSettingValue<bool>("smart-ai");
        auto pit = q.find(key);
        std::array<float, 2> v = {0.f, 0.f};
        if (smart) {
            auto& g = gq[gkey];
            if (pit != q.end()) {
                v = {0.7f * pit->second[0] + 0.3f * g[0], 0.7f * pit->second[1] + 0.3f * g[1]};
            } else {
                v = g;
            }
        } else if (pit != q.end()) {
            v = pit->second;
        }
        // Explore near the place it keeps dying (more the longer it's stuck); greedy before it.
        float eps = x > bestX - 150.f ? std::clamp(0.05f + 0.01f * frontierDeaths, 0.05f, 0.4f) : 0.01f;
        std::uniform_real_distribution<float> u(0.f, 1.f);
        if (u(rng) < eps) return u(rng) < 0.3f ? 1 : 0;
        return v[1] > v[0] ? 1 : 0;
    }

    // Replay the attempt backwards so the death penalty propagates quickly.
    void learn(bool died) {
        if (trajectory.empty()) return;
        float nextMax = 0.f, gNextMax = 0.f;
        for (int i = static_cast<int>(trajectory.size()) - 1; i >= 0; --i) {
            auto& s = trajectory[i];
            bool terminal = i == static_cast<int>(trajectory.size()) - 1;
            float end = died ? -100.f : 200.f;

            auto& v = q[s.key];
            float target = s.reward + (terminal ? end : GAMMA * nextMax);
            v[s.action] += alpha() * (target - v[s.action]);
            nextMax = std::max(v[0], v[1]);

            auto& g = gq[s.gkey];
            float gTarget = s.reward + (terminal ? end : GAMMA * gNextMax);
            g[s.action] += 0.1f * (gTarget - g[s.action]);
            gNextMax = std::max(g[0], g[1]);
        }
        trajectory.clear();
    }

    // ---- persistence ----

    static std::filesystem::path dir() { return Mod::get()->getSaveDir(); }
    std::filesystem::path file(char const* ext) const {
        return dir() / fmt::format("level_{}.{}", levelID, ext);
    }

    template <class T>
    static void put(std::ofstream& f, T const& v) { f.write(reinterpret_cast<const char*>(&v), sizeof(T)); }
    template <class T>
    static bool get(std::ifstream& f, T& v) { return static_cast<bool>(f.read(reinterpret_cast<char*>(&v), sizeof(T))); }

    static void writeInputs(std::filesystem::path const& path, float end, std::vector<Input> const& ins) {
        std::ofstream f(path, std::ios::binary);
        if (!f) return;
        put(f, end);
        for (auto& in : ins) {
            put(f, in.x);
            char h = in.hold ? 1 : 0;
            put(f, h);
        }
    }

    static void readInputs(std::filesystem::path const& path, float& end, std::vector<Input>& ins) {
        ins.clear();
        end = 0.f;
        std::ifstream f(path, std::ios::binary);
        if (!f) return;
        get(f, end);
        Input in;
        char h;
        while (get(f, in.x) && get(f, h)) {
            in.hold = h != 0;
            ins.push_back(in);
        }
    }

    static void writeTable(std::filesystem::path const& path, std::unordered_map<uint64_t, std::array<float, 2>> const& t,
                           float bx, int att) {
        std::ofstream f(path, std::ios::binary);
        if (!f) return;
        put(f, bx);
        put(f, att);
        for (auto& [k, v] : t) {
            put(f, k);
            put(f, v);
        }
    }

    static void readTable(std::filesystem::path const& path, std::unordered_map<uint64_t, std::array<float, 2>>& t,
                          float& bx, int& att) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return;
        get(f, bx);
        get(f, att);
        uint64_t k;
        std::array<float, 2> v;
        while (get(f, k) && get(f, v)) t[k] = v;
    }

    void saveMeta() {
        std::ofstream f(file("meta"), std::ios::binary);
        if (!f) return;
        put(f, trainedSeconds);
        put(f, static_cast<uint32_t>(history.size()));
        for (float h : history) put(f, h);
        put(f, static_cast<uint32_t>(deathMap.size()));
        for (auto& [k, c] : deathMap) {
            put(f, k);
            put(f, c);
        }
    }

    void loadMeta() {
        trainedSeconds = 0.f;
        history.clear();
        deathMap.clear();
        std::ifstream f(file("meta"), std::ios::binary);
        if (!f) return;
        uint32_t n = 0;
        get(f, trainedSeconds);
        if (!get(f, n)) return;
        for (uint32_t i = 0; i < n; ++i) {
            float h;
            if (!get(f, h)) return;
            history.push_back(h);
        }
        if (!get(f, n)) return;
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t k;
            int c;
            if (!get(f, k) || !get(f, c)) return;
            deathMap[k] = c;
        }
    }

    void save() {
        writeInputs(file("run"), bestRunEnd, bestRun);
        writeInputs(file("guide"), guideEnd, guide);
        writeTable(file("qtable"), q, bestX, attempts);
        writeTable(dir() / "general.qtable", gq, 0.f, 0);
        saveMeta();
        if (levelLength > 0) {
            Mod::get()->setSavedValue<float>(fmt::format("best-{}", levelID), pctOf(bestX));
            Mod::get()->setSavedValue<int>(fmt::format("attempts-{}", levelID), attempts);
        }
    }

    void load(int id) {
        if (!gqLoaded) {
            float bx;
            int att;
            readTable(dir() / "general.qtable", gq, bx, att);
            gqLoaded = true;
        }
        if (id == levelID && !q.empty()) return;
        levelID = id;
        q.clear();
        bestX = 0.f;
        attempts = 0;
        frontierDeaths = 0;
        frontierBest = 0.f;
        attemptTimes.clear();
        readInputs(file("run"), bestRunEnd, bestRun);
        readInputs(file("guide"), guideEnd, guide);
        readTable(file("qtable"), q, bestX, attempts);
        loadMeta();
    }

    void resetLevelData() {
        std::error_code ec;
        for (auto ext : {"qtable", "run", "guide", "meta"}) std::filesystem::remove(file(ext), ec);
        Mod::get()->setSavedValue<float>(fmt::format("best-{}", levelID), -1.f);
        Mod::get()->setSavedValue<int>(fmt::format("attempts-{}", levelID), 0);
        int id = levelID;
        levelID = -1;
        q.clear();
        load(id);
    }
};

static Agent g_ai;

static std::filesystem::path exportDir() { return dirs::getGeodeDir() / "ai-export"; }

// Copies every saved AI file between the mod's save folder and a folder the
// user can reach with a file manager.
static int copyAIFiles(std::filesystem::path const& from, std::filesystem::path const& to) {
    std::error_code ec;
    std::filesystem::create_directories(to, ec);
    int n = 0;
    for (auto& entry : std::filesystem::directory_iterator(from, ec)) {
        auto ext = entry.path().extension().string();
        if (ext != ".qtable" && ext != ".run" && ext != ".guide" && ext != ".meta") continue;
        std::filesystem::copy_file(entry.path(), to / entry.path().filename(),
            std::filesystem::copy_options::overwrite_existing, ec);
        if (!ec) n++;
    }
    return n;
}

static void applyTimeScale(bool on) {
    float s = 1.f;
    if (on) {
        s = static_cast<float>(Mod::get()->getSettingValue<double>("training-speed"));
        if (g_ai.rapid) s = std::max(s, static_cast<float>(Mod::get()->getSettingValue<double>("rapid-speed")));
    }
    CCDirector::get()->getScheduler()->setTimeScale(s);
}

static void startAI(PlayLayer* pl) {
    g_ai.load(pl->m_level->m_levelID.value());
    g_ai.levelLength = pl->m_levelLength;
    g_ai.buildObstacles(pl);
    g_ai.active = true;
    g_ai.recording = false;
    g_ai.trajectory.clear();
    g_ai.episodeOver = false;
    g_ai.holding = false;
    g_ai.tick = 0;
    g_ai.lastX = g_ai.startX = pl->m_player1->getPositionX();
    applyTimeScale(true);
}

static void stopAI(PlayLayer* pl) {
    g_ai.active = false;
    pl->handleButton(false, 1, true);
    g_ai.save();
    applyTimeScale(false);
}

static void finishRecording(float endX) {
    g_ai.recording = false;
    g_ai.guideEnd = std::max(0.f, endX);
    g_ai.save();
    Notification::create(fmt::format("Guide saved ({:.0f}% of the level)", g_ai.pctOf(g_ai.guideEnd)),
        NotificationIcon::Success)->show();
}

// ---------- heatmap (drawn in level space, so it scrolls with the level) ----------

constexpr int HEATMAP_TAG = 0xA1EA7;

// set when a replay restarts; the control loop clears its tick accumulator
static bool s_replayAccReset = false;

static void refreshHeatmap(GJBaseGameLayer* gl) {
    if (!gl || !gl->m_objectLayer) return;
    auto node = static_cast<CCDrawNode*>(gl->m_objectLayer->getChildByTag(HEATMAP_TAG));
    if (!node) {
        node = CCDrawNode::create();
        node->setTag(HEATMAP_TAG);
        node->setZOrder(9999);
        gl->m_objectLayer->addChild(node);
    }
    node->clear();
    if (!g_ai.showHeatmap || g_ai.deathMap.empty()) return;
    int maxCount = 1;
    for (auto& [k, c] : g_ai.deathMap) maxCount = std::max(maxCount, c);
    for (auto& [k, c] : g_ai.deathMap) {
        float x = static_cast<float>(k >> 8) * 30.f;
        float y = static_cast<float>(k & 0xFF) * 30.f;
        float a = 0.15f + 0.6f * static_cast<float>(c) / static_cast<float>(maxCount);
        CCPoint verts[4] = {{x, y}, {x + 30.f, y}, {x + 30.f, y + 30.f}, {x, y + 30.f}};
        node->drawPolygon(verts, 4, ccc4f(1.f, 0.1f, 0.1f, a), 0.f, ccc4f(0, 0, 0, 0));
    }
}

// ---------- AI menu (opened from the pause menu) ----------

class AIMenuLayer : public CCLayerColor {
    PauseLayer* m_pause = nullptr;
    CCLabelBMFont* m_stats = nullptr;
    CCMenuItemSpriteExtra* m_startBtn = nullptr;
    CCMenuItemSpriteExtra* m_rapidBtn = nullptr;
    CCMenuItemSpriteExtra* m_rapidNoCpBtn = nullptr;
    CCMenuItemSpriteExtra* m_replayBtn = nullptr;
    CCMenuItemSpriteExtra* m_recordBtn = nullptr;
    CCMenuItemSpriteExtra* m_heatBtn = nullptr;
    CCMenuItemSpriteExtra* m_guideBtn = nullptr;
    CCMenuItemSpriteExtra* m_resetBtn = nullptr;
    CCDrawNode* m_graph = nullptr;
    CCLabelBMFont* m_graphLabel = nullptr;
    bool m_resetArmed = false;
    std::vector<CCNode*> m_hidden;

    static ButtonSprite* makeSprite(const char* text, bool on) {
        return ButtonSprite::create(text, 190, true, "goldFont.fnt", on ? "GJ_button_02.png" : "GJ_button_01.png", 30.f, 0.7f);
    }
    static ButtonSprite* makeSmall(const char* text, bool on) {
        return ButtonSprite::create(text, 80, true, "goldFont.fnt", on ? "GJ_button_02.png" : "GJ_button_04.png", 26.f, 0.55f);
    }

    CCMenuItemSpriteExtra* add(CCMenu* menu, ButtonSprite* spr, SEL_MenuHandler sel, CCPoint pos) {
        auto btn = CCMenuItemSpriteExtra::create(spr, this, sel);
        btn->setPosition(pos);
        menu->addChild(btn);
        return btn;
    }

public:
    static AIMenuLayer* create(PauseLayer* pause) {
        auto ret = new AIMenuLayer();
        if (ret->init(pause)) {
            ret->autorelease();
            return ret;
        }
        delete ret;
        return nullptr;
    }

    bool init(PauseLayer* pause) {
        if (!CCLayerColor::initWithColor({0, 0, 0, 160})) return false;
        m_pause = pause;
        auto win = CCDirector::get()->getWinSize();
        CCPoint c = {win.width / 2, win.height / 2};

        // Hide the pause menu underneath so its buttons can't be pressed through us.
        for (auto child : CCArrayExt<CCNode*>(pause->getChildren())) {
            if (child->isVisible()) {
                child->setVisible(false);
                m_hidden.push_back(child);
            }
        }

        auto bg = CCScale9Sprite::create("GJ_square01.png");
        bg->setContentSize({460.f, 300.f});
        bg->setPosition(c);
        this->addChild(bg);

        auto title = CCLabelBMFont::create("AI Learner", "goldFont.fnt");
        title->setScale(0.8f);
        title->setPosition({c.x, c.y + 130.f});
        this->addChild(title);

        m_stats = CCLabelBMFont::create("", "chatFont.fnt");
        m_stats->setScale(0.6f);
        m_stats->setPosition({c.x, c.y + 108.f});
        this->addChild(m_stats);

        // progress graph on the right
        m_graphLabel = CCLabelBMFont::create("Progress per attempt", "chatFont.fnt");
        m_graphLabel->setScale(0.55f);
        m_graphLabel->setPosition({c.x + 115.f, c.y + 88.f});
        this->addChild(m_graphLabel);
        m_graph = CCDrawNode::create();
        this->addChild(m_graph);

        auto menu = CCMenu::create();
        menu->setPosition(c);
        this->addChild(menu);

        m_startBtn = add(menu, makeSprite("Start AI", false), menu_selector(AIMenuLayer::onStart), {-110.f, 75.f});
        m_rapidBtn = add(menu, makeSprite("Learn Rapidly", false), menu_selector(AIMenuLayer::onRapid), {-110.f, 42.f});
        m_rapidNoCpBtn = add(menu, makeSprite("Rapid (No Checkpoints)", false), menu_selector(AIMenuLayer::onRapidNoCp), {-110.f, 9.f});
        m_replayBtn = add(menu, makeSprite("Replay Best Run", false), menu_selector(AIMenuLayer::onReplay), {-110.f, -24.f});
        m_recordBtn = add(menu, makeSprite("Record Guide", false), menu_selector(AIMenuLayer::onRecord), {-110.f, -57.f});

        m_heatBtn = add(menu, makeSmall("Heatmap", false), menu_selector(AIMenuLayer::onHeatmap), {-176.f, -97.f});
        m_guideBtn = add(menu, makeSmall("Guide", false), menu_selector(AIMenuLayer::onGuide), {-88.f, -97.f});
        m_resetBtn = add(menu, makeSmall("Reset", false), menu_selector(AIMenuLayer::onReset), {0.f, -97.f});
        add(menu, makeSmall("Export", false), menu_selector(AIMenuLayer::onExport), {88.f, -97.f});
        add(menu, makeSmall("Import", false), menu_selector(AIMenuLayer::onImport), {176.f, -97.f});

        add(menu, ButtonSprite::create("Close", "goldFont.fnt", "GJ_button_04.png", 0.6f),
            menu_selector(AIMenuLayer::onClose), {0.f, -128.f});

        this->refresh();
        return true;
    }

    void drawGraph() {
        auto win = CCDirector::get()->getWinSize();
        float x0 = win.width / 2 + 20.f, y0 = win.height / 2 - 70.f, w = 190.f, h = 150.f;
        m_graph->clear();
        CCPoint box[4] = {{x0, y0}, {x0 + w, y0}, {x0 + w, y0 + h}, {x0, y0 + h}};
        m_graph->drawPolygon(box, 4, ccc4f(0, 0, 0, 0.35f), 1.f, ccc4f(1, 1, 1, 0.3f));
        for (int pct : {25, 50, 75}) {
            float y = y0 + h * pct / 100.f;
            m_graph->drawSegment({x0, y}, {x0 + w, y}, 0.3f, ccc4f(1, 1, 1, 0.12f));
        }
        auto& hist = g_ai.history;
        if (hist.size() < 2) return;
        // running best (yellow) and each attempt (cyan)
        float best = 0.f;
        size_t n = hist.size();
        for (size_t i = 1; i < n; ++i) {
            float xa = x0 + w * (i - 1) / (n - 1), xb = x0 + w * i / (n - 1);
            m_graph->drawSegment({xa, y0 + h * hist[i - 1] / 100.f}, {xb, y0 + h * hist[i] / 100.f}, 0.6f,
                ccc4f(0.2f, 0.85f, 1.f, 0.8f));
            float prevBest = std::max(best, hist[i - 1]);
            best = std::max(prevBest, hist[i]);
            m_graph->drawSegment({xa, y0 + h * prevBest / 100.f}, {xb, y0 + h * best / 100.f}, 0.8f,
                ccc4f(1.f, 0.85f, 0.2f, 0.9f));
        }
    }

    void refresh() {
        auto pl = PlayLayer::get();
        if (pl && g_ai.levelLength <= 0) g_ai.levelLength = pl->m_levelLength;
        int mins = static_cast<int>(g_ai.trainedSeconds / 60.f);
        m_stats->setString(fmt::format("Attempts {}  |  Best {:.1f}%  |  Best run {:.1f}%  |  Trained {}m  |  {:.0f} att/min",
            g_ai.attempts, g_ai.pctOf(g_ai.bestX), g_ai.pctOf(g_ai.bestRunEnd), mins, g_ai.attemptsPerMinute()).c_str());
        m_startBtn->setNormalImage(makeSprite(g_ai.active ? "Stop AI" : "Start AI", g_ai.active));
        bool withCp = (g_ai.rapid || g_ai.verifying) && g_ai.useCheckpoints;
        bool noCp = g_ai.rapid && !g_ai.useCheckpoints;
        m_rapidBtn->setNormalImage(makeSprite(withCp ? "Learn Rapidly: ON" : "Learn Rapidly", withCp));
        m_rapidNoCpBtn->setNormalImage(makeSprite(noCp ? "No Checkpoints: ON" : "Rapid (No Checkpoints)", noCp));
        m_recordBtn->setNormalImage(makeSprite(g_ai.recording ? "Stop Recording" : "Record Guide", g_ai.recording));
        m_heatBtn->setNormalImage(makeSmall(g_ai.showHeatmap ? "Heatmap ON" : "Heatmap", g_ai.showHeatmap));
        bool guideOn = g_ai.useGuide && !g_ai.guide.empty();
        m_guideBtn->setNormalImage(makeSmall(g_ai.guide.empty() ? "No Guide" : guideOn ? "Guide ON" : "Guide OFF", guideOn));
        m_resetBtn->setNormalImage(makeSmall(m_resetArmed ? "Sure?" : "Reset", m_resetArmed));
        this->drawGraph();
    }

    void onStart(CCObject*) {
        auto pl = PlayLayer::get();
        if (!pl) return;
        if (g_ai.active) {
            stopAI(pl);
            this->refresh();
        } else {
            startAI(pl);
            this->resumeGame();
        }
    }

    // Turning rapid on also starts the AI and resumes, so one press gets it going.
    void onRapid(CCObject*) { this->toggleRapid(true); }
    void onRapidNoCp(CCObject*) { this->toggleRapid(false); }

    void toggleRapid(bool checkpoints) {
        auto pl = PlayLayer::get();
        if (!pl) return;
        bool wasOn = (g_ai.rapid || g_ai.verifying) && g_ai.useCheckpoints == checkpoints;
        g_ai.rapid = !wasOn;
        g_ai.useCheckpoints = checkpoints;
        g_ai.verifying = false;
        g_ai.pendingSwitch = Agent::Switch::None;
        g_ai.checkpointXs.clear();
        g_ai.deathsSinceProgress = 0;
        bool wantPractice = g_ai.rapid && checkpoints;
        if (wantPractice != pl->m_isPracticeMode) pl->togglePracticeMode(wantPractice);
        if (g_ai.rapid && !g_ai.active) {
            startAI(pl);
            this->resumeGame();
            return;
        }
        applyTimeScale(g_ai.active);
        this->refresh();
    }

    void onReplay(CCObject*) {
        auto pl = PlayLayer::get();
        if (!pl) return;
        g_ai.load(pl->m_level->m_levelID.value());
        if (g_ai.bestRun.empty()) {
            Notification::create("No best run recorded yet", NotificationIcon::Info)->show();
            return;
        }
        if (g_ai.active) stopAI(pl);
        g_ai.rapid = false;
        g_ai.verifying = false;
        g_ai.recording = false;
        g_ai.pendingSwitch = Agent::Switch::None;
        g_ai.replaying = true;
        g_ai.replayIdx = 0;
        applyTimeScale(false);
        this->resumeGame();
        // restart from 0% in normal mode (togglePracticeMode resets the level itself)
        if (pl->m_isPracticeMode) pl->togglePracticeMode(false);
        else pl->resetLevel();
    }

    // Record Guide: the player plays from 0%; their inputs become the AI's guide.
    void onRecord(CCObject*) {
        auto pl = PlayLayer::get();
        if (!pl) return;
        if (g_ai.recording) {
            finishRecording(pl->m_player1->getPositionX());
            this->refresh();
            return;
        }
        g_ai.load(pl->m_level->m_levelID.value());
        g_ai.levelLength = pl->m_levelLength;
        if (g_ai.active) stopAI(pl);
        g_ai.rapid = false;
        g_ai.verifying = false;
        g_ai.replaying = false;
        g_ai.pendingSwitch = Agent::Switch::None;
        g_ai.guide.clear();
        g_ai.guideEnd = 0.f;
        g_ai.recording = true;
        applyTimeScale(false);
        Notification::create("Recording: play the start of the level", NotificationIcon::Info)->show();
        this->resumeGame();
        if (pl->m_isPracticeMode) pl->togglePracticeMode(false);
        else pl->resetLevel();
    }

    void onHeatmap(CCObject*) {
        auto pl = PlayLayer::get();
        if (!pl) return;
        g_ai.load(pl->m_level->m_levelID.value());
        g_ai.showHeatmap = !g_ai.showHeatmap;
        refreshHeatmap(pl);
        this->refresh();
    }

    void onGuide(CCObject*) {
        g_ai.useGuide = !g_ai.useGuide;
        this->refresh();
    }

    void onReset(CCObject*) {
        auto pl = PlayLayer::get();
        if (!pl) return;
        if (!m_resetArmed) {
            m_resetArmed = true;
            this->refresh();
            return;
        }
        m_resetArmed = false;
        if (g_ai.active) stopAI(pl);
        g_ai.load(pl->m_level->m_levelID.value());
        g_ai.resetLevelData();
        refreshHeatmap(pl);
        Notification::create("AI data for this level was reset", NotificationIcon::Success)->show();
        this->refresh();
    }

    void onExport(CCObject*) {
        g_ai.save();
        int n = copyAIFiles(Agent::dir(), exportDir());
        Notification::create(fmt::format("Exported {} files to geode/ai-export", n), NotificationIcon::Success)->show();
    }

    void onImport(CCObject*) {
        int n = copyAIFiles(exportDir(), Agent::dir());
        int id = g_ai.levelID;
        g_ai.levelID = -1;
        g_ai.gqLoaded = false;
        g_ai.gq.clear();
        g_ai.load(id);
        Notification::create(n ? fmt::format("Imported {} files", n) : std::string("Nothing found in geode/ai-export"),
            n ? NotificationIcon::Success : NotificationIcon::Warning)->show();
        this->refresh();
    }

    void onClose(CCObject*) {
        for (auto node : m_hidden) node->setVisible(true);
        this->removeFromParent();
    }

    void resumeGame() {
        auto pause = m_pause;
        this->onClose(nullptr);
        pause->onResume(nullptr);
    }
};

// ---------- pause menu button ----------

class $modify(AIPauseLayer, PauseLayer) {
    void customSetup() {
        PauseLayer::customSetup();
        // Our own menu, so the button doesn't depend on the node-ids mod being installed.
        auto spr = ButtonSprite::create(g_ai.active ? "AI (running)" : g_ai.recording ? "AI (recording)" : "AI Menu",
            "goldFont.fnt", "GJ_button_01.png", 0.8f);
        auto btn = CCMenuItemSpriteExtra::create(spr, this, menu_selector(AIPauseLayer::onOpenAIMenu));
        btn->setID("ai-menu-button"_spr);
        auto menu = CCMenu::create();
        menu->setID("ai-learn-menu"_spr);
        menu->addChild(btn);
        auto win = CCDirector::get()->getWinSize();
        menu->setPosition({win.width - spr->getContentSize().width / 2 - 10.f, 30.f});
        menu->setZOrder(100);
        this->addChild(menu);
    }

    void onOpenAIMenu(CCObject*) {
        auto pl = PlayLayer::get();
        if (!pl) return;
        g_ai.load(pl->m_level->m_levelID.value());
        g_ai.levelLength = pl->m_levelLength;
        auto layer = AIMenuLayer::create(this);
        layer->setZOrder(1000);
        this->addChild(layer);
    }
};

// ---------- episode lifecycle ----------

class $modify(AIPlayLayer, PlayLayer) {
    struct Fields {
        CCLabelBMFont* label = nullptr;
    };

    void resetLevel() {
        PlayLayer::resetLevel();
        if (g_ai.replaying) {
            g_ai.replayIdx = 0;
            s_replayAccReset = true;
            handleButton(false, 1, true);
            return;
        }
        if (g_ai.recording) {
            // a new attempt while recording restarts the guide from scratch
            g_ai.guide.clear();
            return;
        }
        if (!g_ai.active) return;
        g_ai.attemptInputs.clear();
        g_ai.attempts++;
        g_ai.episodeOver = false;
        g_ai.deadTicks = 0;
        g_ai.holding = false;
        g_ai.tick = 0;
        g_ai.trajectory.clear();
        g_ai.lastX = g_ai.startX = m_player1->getPositionX();
        if (g_ai.attempts % 25 == 0) g_ai.save();
    }

    void destroyPlayer(PlayerObject* player, GameObject* obj) {
        bool realDeath = g_ai.active && !g_ai.episodeOver && obj != m_anticheatSpike;
        bool recordDeath = g_ai.recording && obj != m_anticheatSpike;
        PlayLayer::destroyPlayer(player, obj);
        if (!player->m_isDead) return;
        if (recordDeath) {
            // keep the guide up to a little before the mistake
            finishRecording(player->getPositionX() - 90.f);
        }
        if (g_ai.replaying && obj != m_anticheatSpike) {
            g_ai.replaying = false;
            handleButton(false, 1, true);
        }
        if (realDeath) {
            g_ai.episodeOver = true;
            g_ai.recordAttemptEnd(player->getPositionX(), player->getPositionY(), true);
            g_ai.commitAttempt(player->getPositionX());
            g_ai.learn(true);
            handleButton(false, 1, true);
            if (g_ai.showHeatmap) refreshHeatmap(this);
            if (g_ai.verifying) {
                g_ai.pendingSwitch = Agent::Switch::ToPractice;
            }
            if (g_ai.rapid && g_ai.useCheckpoints) {
                if (g_ai.bestX > g_ai.bestAtLastCheck + 30.f) {
                    g_ai.bestAtLastCheck = g_ai.bestX;
                    g_ai.deathsSinceProgress = 0;
                } else if (++g_ai.deathsSinceProgress > 40 && !g_ai.checkpointXs.empty()) {
                    // The last checkpoint may be in a doomed spot (e.g. mid-jump): drop it.
                    this->removeCheckpoint(false);
                    g_ai.checkpointXs.pop_back();
                    g_ai.deathsSinceProgress = 0;
                }
            }
        }
    }

    // In rapid mode we respawn ourselves right after death; skip the game's own
    // delayed reset that would otherwise restart the new attempt a second time.
    void delayedResetLevel() {
        if (g_ai.skipDelayedReset) {
            g_ai.skipDelayedReset = false;
            return;
        }
        PlayLayer::delayedResetLevel();
    }

    void levelComplete() {
        if (g_ai.recording) {
            finishRecording(m_levelLength);
            PlayLayer::levelComplete();
            return;
        }
        if (g_ai.replaying) {
            g_ai.replaying = false;
            PlayLayer::levelComplete();
            return;
        }
        if (g_ai.active && !g_ai.episodeOver) {
            g_ai.commitAttempt(m_levelLength + 100.f);
            g_ai.recordAttemptEnd(m_levelLength, 0.f, false);
        }
        if (g_ai.active && g_ai.rapid && m_isPracticeMode && !g_ai.episodeOver) {
            g_ai.episodeOver = true;
            g_ai.learn(false);
            g_ai.save();
            g_ai.pendingSwitch = Agent::Switch::ToNormal;
            return;
        }
        if (g_ai.active && (g_ai.verifying || (g_ai.rapid && !m_isPracticeMode))) {
            // Beat it for real: stop and let the normal end screen play.
            g_ai.verifying = false;
            g_ai.rapid = false;
            g_ai.active = false;
            g_ai.learn(false);
            g_ai.save();
            applyTimeScale(false);
            PlayLayer::levelComplete();
            return;
        }
        if (g_ai.active && !g_ai.episodeOver) {
            g_ai.episodeOver = true;
            g_ai.learn(false);
            g_ai.save();
        }
        PlayLayer::levelComplete();
    }

    void onQuit() {
        g_ai.replaying = false;
        if (g_ai.recording) finishRecording(m_player1->getPositionX());
        if (g_ai.active) {
            g_ai.save();
            g_ai.active = false;
            g_ai.rapid = false;
            g_ai.verifying = false;
            g_ai.pendingSwitch = Agent::Switch::None;
            g_ai.checkpointXs.clear();
            applyTimeScale(false);
        }
        PlayLayer::onQuit();
    }

    void updateAILabel() {
        auto f = m_fields.self();
        if (!f->label) {
            f->label = CCLabelBMFont::create("", "bigFont.fnt");
            f->label->setScale(0.4f);
            f->label->setAnchorPoint({0.f, 1.f});
            auto win = CCDirector::get()->getWinSize();
            f->label->setPosition({8.f, win.height - 8.f});
            f->label->setZOrder(1000);
            this->addChild(f->label);
        }
        f->label->setVisible(g_ai.active || g_ai.replaying || g_ai.recording);
        if (g_ai.recording) {
            f->label->setString(fmt::format("Recording guide... {:.1f}%", g_ai.pctOf(m_player1->getPositionX())).c_str());
        } else if (g_ai.replaying) {
            f->label->setString(fmt::format("Replaying best run ({:.1f}%)", g_ai.pctOf(g_ai.bestRunEnd)).c_str());
        } else if (g_ai.active) {
            const char* phase = g_ai.verifying ? "normal run"
                : g_ai.rapid ? (g_ai.useCheckpoints ? "practice (rapid)" : "rapid, no checkpoints") : "learning";
            bool guided = g_ai.useGuide && !g_ai.guide.empty() && m_player1->getPositionX() < g_ai.guideEnd;
            f->label->setString(fmt::format("AI {}{}  attempt {}  best {:.1f}%", phase, guided ? " (guided)" : "",
                g_ai.attempts, g_ai.pctOf(g_ai.bestX)).c_str());
        }
    }
};

// ---------- the control loop ----------

// GD 2.2 simulates physics at a fixed 240 ticks per second.
constexpr float PHYSICS_TPS = 240.f;
// a hair over one tick so float rounding never yields a 0-tick update
constexpr float TICK_DT = 1.0001f / PHYSICS_TPS;
static float s_replayAcc = 0.f;

class $modify(AIGameLayer, GJBaseGameLayer) {
    // Record the player's own inputs while recording a guide.
    void handleButton(bool down, int button, bool isPlayer1) {
        GJBaseGameLayer::handleButton(down, button, isPlayer1);
        auto pl = PlayLayer::get();
        if (g_ai.recording && button == 1 && isPlayer1 && pl && static_cast<GJBaseGameLayer*>(pl) == this) {
            g_ai.guide.push_back({m_player1->getPositionX(), down});
        }
    }

    void update(float dt) {
        auto pl = PlayLayer::get();
        bool isPlay = pl && static_cast<GJBaseGameLayer*>(pl) == this;
        if (isPlay && g_ai.replaying) {
            // One physics tick per update call, applying recorded inputs between
            // ticks, so presses land on exactly the same tick as when recorded.
            if (s_replayAccReset) {
                s_replayAcc = 0.f;
                s_replayAccReset = false;
            }
            s_replayAcc += dt * PHYSICS_TPS;
            int ticks = 0;
            while (s_replayAcc >= 1.f && ticks < 64 && g_ai.replaying) {
                s_replayAcc -= 1.f;
                ticks++;
                this->replayStep();
                GJBaseGameLayer::update(TICK_DT);
            }
            static_cast<AIPlayLayer*>(pl)->updateAILabel();
            return;
        }
        if (isPlay && g_ai.recording) {
            GJBaseGameLayer::update(dt);
            static_cast<AIPlayLayer*>(pl)->updateAILabel();
            return;
        }
        if (!g_ai.active || !isPlay) {
            GJBaseGameLayer::update(dt);
            return;
        }

        // Rapid mode runs several game ticks per drawn frame; only the last one is seen.
        int steps = g_ai.rapid ? static_cast<int>(Mod::get()->getSettingValue<int64_t>("rapid-steps")) : 1;
        for (int i = 0; i < steps && g_ai.active; ++i) {
            GJBaseGameLayer::update(dt);
            this->aiStep(pl);
        }
        g_ai.trainedSeconds += dt * steps;   // dt already includes the game speed
        static_cast<AIPlayLayer*>(pl)->updateAILabel();
    }

    void replayStep() {
        if (m_player1->m_isDead) return;
        // inputs recorded at x were applied before the tick that starts at x
        float x = m_player1->getPositionX();
        auto& run = g_ai.bestRun;
        while (g_ai.replayIdx < run.size() && run[g_ai.replayIdx].x <= x) {
            handleButton(run[g_ai.replayIdx].hold, 1, true);
            g_ai.replayIdx++;
        }
    }

    void aiStep(PlayLayer* pl) {
        if (g_ai.pendingSwitch != Agent::Switch::None) {
            auto to = g_ai.pendingSwitch;
            g_ai.pendingSwitch = Agent::Switch::None;
            g_ai.rapid = to == Agent::Switch::ToPractice;
            g_ai.verifying = to == Agent::Switch::ToNormal;
            g_ai.checkpointXs.clear();
            g_ai.deathsSinceProgress = 0;
            handleButton(false, 1, true);
            g_ai.holding = false;
            applyTimeScale(true);
            // togglePracticeMode restarts the level in the new mode; skip the
            // game's own delayed respawn if we got here from a death.
            if (m_player1->m_isDead) g_ai.skipDelayedReset = true;
            pl->togglePracticeMode(g_ai.rapid);
            return;
        }
        if (g_ai.episodeOver || m_player1->m_isDead) {
            // Rapid mode: skip the death animation and respawn after a few ticks.
            if (g_ai.rapid && m_player1->m_isDead && ++g_ai.deadTicks == 4) {
                g_ai.skipDelayedReset = true;
                pl->resetLevel();
            }
            return;
        }

        auto p = m_player1;
        float x = p->getPositionX();
        g_ai.bestX = std::max(g_ai.bestX, x);

        // Drop a checkpoint on solid ground a little behind the furthest point reached.
        if (g_ai.rapid && g_ai.useCheckpoints && p->m_isOnGround) {
            float lastCp = g_ai.checkpointXs.empty() ? 0.f : g_ai.checkpointXs.back();
            if (x > lastCp + 450.f && x < g_ai.bestX - 150.f) {
                pl->markCheckpoint();
                g_ai.checkpointXs.push_back(x);
            }
        }

        if (++g_ai.tick % Agent::DECIDE_EVERY != 0) return;

        // reward = forward progress since the last decision
        if (!g_ai.trajectory.empty()) {
            g_ai.trajectory.back().reward = (x - g_ai.lastX) * 0.1f;
        }
        g_ai.lastX = x;

        uint64_t key = Agent::makeKey(p, g_ai.holding);
        uint64_t gkey = g_ai.makeFeatureKey(p, g_ai.holding);
        int a;
        if (g_ai.useGuide && !g_ai.guide.empty() && x < g_ai.guideEnd) {
            // Follow the player's guide here; the AI still learns from these steps.
            a = g_ai.guideHoldAt(x) ? 1 : 0;
        } else {
            a = g_ai.choose(key, gkey, x);
        }
        g_ai.trajectory.push_back({key, gkey, a, 0.f});

        bool hold = a == 1;
        if (hold != g_ai.holding) {
            handleButton(hold, 1, true);
            g_ai.holding = hold;
            g_ai.attemptInputs.push_back({x, hold});
        }
    }
};
