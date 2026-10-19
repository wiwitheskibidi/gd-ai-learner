// GD AI Learner - a Geode mod that adds an "AI" button to the pause menu.
// Press it and a Q-learning agent takes over the level you are on, dying,
// respawning and improving every attempt. Press it again to stop.

#include <Geode/Geode.hpp>
#include <Geode/modify/PauseLayer.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/GJBaseGameLayer.hpp>
#include <unordered_map>
#include <array>
#include <vector>
#include <fstream>
#include <random>

using namespace geode::prelude;

// ---------- the learner ----------

struct Step {
    uint64_t key;
    int action;
    float reward;
};

struct Agent {
    bool active = false;
    bool episodeOver = false;
    int levelID = 0;
    int attempts = 0;
    float bestX = 0.f;
    float startX = 0.f;
    float lastX = 0.f;
    bool holding = false;
    int tick = 0;

    // "Learn Rapidly": faster game speed, instant respawn, bigger learning steps.
    bool rapid = false;
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

    std::unordered_map<uint64_t, std::array<float, 2>> q;
    std::vector<Step> trajectory;
    std::mt19937 rng{std::random_device{}()};

    static constexpr int DECIDE_EVERY = 3;   // physics ticks between decisions
    float alpha() const { return rapid ? 0.5f : 0.3f; }
    static constexpr float GAMMA = 0.98f;

    // State = where we are in the level + vertical position/velocity + gamemode.
    // Levels are deterministic, so position-based states let the agent learn
    // exactly where to click on this level.
    static uint64_t makeKey(PlayerObject* p, bool holding) {
        int xb = static_cast<int>(p->getPositionX() / 10.f);
        int yb = static_cast<int>(p->getPositionY() / 15.f);
        double vy = p->m_yVelocity;
        int vb = vy > 2 ? 2 : (vy < -2 ? 0 : 1);
        int mode = p->m_isShip ? 1 : p->m_isBall ? 2 : p->m_isBird ? 3 : p->m_isDart ? 4
                 : p->m_isRobot ? 5 : p->m_isSpider ? 6 : p->m_isSwing ? 7 : 0;
        uint64_t k = static_cast<uint32_t>(xb);
        k = (k << 12) | (static_cast<uint32_t>(yb + 100) & 0xFFF);
        k = (k << 2) | vb;
        k = (k << 3) | mode;
        k = (k << 1) | (p->m_isUpsideDown ? 1 : 0);
        k = (k << 1) | (holding ? 1 : 0);
        return k;
    }

    int choose(uint64_t key, float x) {
        auto& v = q[key];
        // Explore mostly near the place we keep dying; play greedily before it.
        float eps = x > bestX - 150.f ? std::max(0.05f, 0.5f - attempts * 0.002f) : 0.01f;
        std::uniform_real_distribution<float> u(0.f, 1.f);
        if (u(rng) < eps) return u(rng) < 0.3f ? 1 : 0;
        return v[1] > v[0] ? 1 : 0;
    }

    // Replay the attempt backwards so the death penalty propagates quickly.
    void learn(bool died) {
        if (trajectory.empty()) return;
        float nextMax = 0.f;
        for (int i = static_cast<int>(trajectory.size()) - 1; i >= 0; --i) {
            auto& s = trajectory[i];
            bool terminal = i == static_cast<int>(trajectory.size()) - 1;
            float target = s.reward + (terminal ? (died ? -100.f : 200.f) : GAMMA * nextMax);
            auto& v = q[s.key];
            v[s.action] += alpha() * (target - v[s.action]);
            nextMax = std::max(v[0], v[1]);
        }
        trajectory.clear();
    }

    std::filesystem::path file() const {
        return Mod::get()->getSaveDir() / fmt::format("level_{}.qtable", levelID);
    }

    void save() {
        std::ofstream f(file(), std::ios::binary);
        if (!f) return;
        f.write(reinterpret_cast<const char*>(&bestX), sizeof(bestX));
        f.write(reinterpret_cast<const char*>(&attempts), sizeof(attempts));
        for (auto& [k, v] : q) {
            f.write(reinterpret_cast<const char*>(&k), sizeof(k));
            f.write(reinterpret_cast<const char*>(v.data()), sizeof(float) * 2);
        }
    }

    void load(int id) {
        if (id == levelID && !q.empty()) return;
        levelID = id;
        q.clear();
        bestX = 0.f;
        attempts = 0;
        std::ifstream f(file(), std::ios::binary);
        if (!f) return;
        f.read(reinterpret_cast<char*>(&bestX), sizeof(bestX));
        f.read(reinterpret_cast<char*>(&attempts), sizeof(attempts));
        uint64_t k;
        std::array<float, 2> v;
        while (f.read(reinterpret_cast<char*>(&k), sizeof(k)) &&
               f.read(reinterpret_cast<char*>(v.data()), sizeof(float) * 2)) {
            q[k] = v;
        }
    }
};

static Agent g_ai;

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
    g_ai.active = true;
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

// ---------- AI menu (opened from the pause menu) ----------

class AIMenuLayer : public CCLayerColor {
    PauseLayer* m_pause = nullptr;
    CCLabelBMFont* m_stats = nullptr;
    CCMenuItemSpriteExtra* m_startBtn = nullptr;
    CCMenuItemSpriteExtra* m_rapidBtn = nullptr;
    std::vector<CCNode*> m_hidden;

    static ButtonSprite* makeSprite(const char* text, bool on) {
        return ButtonSprite::create(text, 200, true, "goldFont.fnt", on ? "GJ_button_02.png" : "GJ_button_01.png", 32.f, 0.8f);
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

        // Hide the pause menu underneath so its buttons can't be pressed through us.
        for (auto child : CCArrayExt<CCNode*>(pause->getChildren())) {
            if (child->isVisible()) {
                child->setVisible(false);
                m_hidden.push_back(child);
            }
        }

        auto bg = CCScale9Sprite::create("GJ_square01.png");
        bg->setContentSize({300.f, 230.f});
        bg->setPosition(win / 2);
        this->addChild(bg);

        auto title = CCLabelBMFont::create("AI Learner", "goldFont.fnt");
        title->setPosition({win.width / 2, win.height / 2 + 90.f});
        this->addChild(title);

        m_stats = CCLabelBMFont::create("", "chatFont.fnt");
        m_stats->setPosition({win.width / 2, win.height / 2 + 60.f});
        this->addChild(m_stats);

        auto menu = CCMenu::create();
        menu->setPosition(win / 2);
        this->addChild(menu);

        m_startBtn = CCMenuItemSpriteExtra::create(makeSprite("Start AI", false), this, menu_selector(AIMenuLayer::onStart));
        m_startBtn->setPosition({0.f, 20.f});
        menu->addChild(m_startBtn);

        m_rapidBtn = CCMenuItemSpriteExtra::create(makeSprite("Learn Rapidly", false), this, menu_selector(AIMenuLayer::onRapid));
        m_rapidBtn->setPosition({0.f, -25.f});
        menu->addChild(m_rapidBtn);

        auto closeBtn = CCMenuItemSpriteExtra::create(
            ButtonSprite::create("Close", "goldFont.fnt", "GJ_button_04.png", 0.7f), this, menu_selector(AIMenuLayer::onClose));
        closeBtn->setPosition({0.f, -80.f});
        menu->addChild(closeBtn);

        this->refresh();
        return true;
    }

    void refresh() {
        auto pl = PlayLayer::get();
        float pct = pl && pl->m_levelLength > 0 ? 100.f * g_ai.bestX / pl->m_levelLength : 0.f;
        m_stats->setString(fmt::format("Attempts: {}   Best: {:.1f}%   States: {}",
            g_ai.attempts, std::min(pct, 100.f), g_ai.q.size()).c_str());
        m_startBtn->setNormalImage(makeSprite(g_ai.active ? "Stop AI" : "Start AI", g_ai.active));
        bool cycling = g_ai.rapid || g_ai.verifying;
        m_rapidBtn->setNormalImage(makeSprite(cycling ? "Learn Rapidly: ON" : "Learn Rapidly", cycling));
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
    void onRapid(CCObject*) {
        auto pl = PlayLayer::get();
        if (!pl) return;
        g_ai.rapid = !(g_ai.rapid || g_ai.verifying);
        g_ai.verifying = false;
        g_ai.pendingSwitch = Agent::Switch::None;
        g_ai.checkpointXs.clear();
        g_ai.deathsSinceProgress = 0;
        if (g_ai.rapid != pl->m_isPracticeMode) pl->togglePracticeMode(g_ai.rapid);
        if (g_ai.rapid && !g_ai.active) {
            startAI(pl);
            this->resumeGame();
            return;
        }
        applyTimeScale(g_ai.active);
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
        auto spr = ButtonSprite::create(g_ai.active ? "AI (running)" : "AI Menu", "goldFont.fnt", "GJ_button_01.png", 0.8f);
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
        if (!PlayLayer::get()) return;
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
        if (!g_ai.active) return;
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
        PlayLayer::destroyPlayer(player, obj);
        if (realDeath && player->m_isDead) {
            g_ai.episodeOver = true;
            g_ai.learn(true);
            handleButton(false, 1, true);
            if (g_ai.verifying) {
                g_ai.pendingSwitch = Agent::Switch::ToPractice;
            }
            if (g_ai.rapid) {
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
        if (g_ai.active && g_ai.rapid && m_isPracticeMode && !g_ai.episodeOver) {
            g_ai.episodeOver = true;
            g_ai.learn(false);
            g_ai.save();
            g_ai.pendingSwitch = Agent::Switch::ToNormal;
            return;
        }
        if (g_ai.active && g_ai.verifying) {
            // Beat it for real: stop and let the normal end screen play.
            g_ai.verifying = false;
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
        f->label->setVisible(g_ai.active);
        if (g_ai.active) {
            float pct = m_levelLength > 0 ? 100.f * g_ai.bestX / m_levelLength : 0.f;
            const char* phase = g_ai.verifying ? "normal run" : g_ai.rapid ? "practice (rapid)" : "learning";
            f->label->setString(fmt::format("AI {}  attempt {}  best {:.1f}%", phase, g_ai.attempts, std::min(pct, 100.f)).c_str());
        }
    }
};

// ---------- the control loop ----------

class $modify(AIGameLayer, GJBaseGameLayer) {
    void update(float dt) {
        auto pl = PlayLayer::get();
        bool ours = g_ai.active && pl && static_cast<GJBaseGameLayer*>(pl) == this;
        if (!ours) {
            GJBaseGameLayer::update(dt);
            return;
        }

        // Rapid mode runs several game ticks per drawn frame; only the last one is seen.
        int steps = g_ai.rapid ? static_cast<int>(Mod::get()->getSettingValue<int64_t>("rapid-steps")) : 1;
        for (int i = 0; i < steps && g_ai.active; ++i) {
            GJBaseGameLayer::update(dt);
            this->aiStep(pl);
        }
        static_cast<AIPlayLayer*>(pl)->updateAILabel();
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
        if (g_ai.rapid && p->m_isOnGround) {
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
        int a = g_ai.choose(key, x);
        g_ai.trajectory.push_back({key, a, 0.f});

        bool hold = a == 1;
        if (hold != g_ai.holding) {
            handleButton(hold, 1, true);
            g_ai.holding = hold;
        }
    }
};
