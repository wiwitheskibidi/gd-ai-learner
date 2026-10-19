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

    std::unordered_map<uint64_t, std::array<float, 2>> q;
    std::vector<Step> trajectory;
    std::mt19937 rng{std::random_device{}()};

    static constexpr int DECIDE_EVERY = 3;   // physics ticks between decisions
    static constexpr float ALPHA = 0.3f;
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
            v[s.action] += ALPHA * (target - v[s.action]);
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
    float s = on ? static_cast<float>(Mod::get()->getSettingValue<double>("training-speed")) : 1.f;
    CCDirector::get()->getScheduler()->setTimeScale(s);
}

// ---------- pause menu button ----------

class $modify(AIPauseLayer, PauseLayer) {
    void customSetup() {
        PauseLayer::customSetup();
        auto menu = this->getChildByID("right-button-menu");
        if (!menu) return;
        auto spr = ButtonSprite::create(g_ai.active ? "Stop AI" : "AI Learn", "goldFont.fnt", "GJ_button_01.png", 0.7f);
        auto btn = CCMenuItemSpriteExtra::create(spr, this, menu_selector(AIPauseLayer::onToggleAI));
        btn->setID("ai-learn-button"_spr);
        menu->addChild(btn);
        menu->updateLayout();
    }

    void onToggleAI(CCObject*) {
        auto pl = PlayLayer::get();
        if (!pl) return;
        g_ai.active = !g_ai.active;
        if (g_ai.active) {
            g_ai.load(pl->m_level->m_levelID.value());
            g_ai.trajectory.clear();
            g_ai.episodeOver = false;
            g_ai.holding = false;
            g_ai.tick = 0;
            g_ai.lastX = g_ai.startX = pl->m_player1->getPositionX();
        } else {
            pl->handleButton(false, 1, true);
            g_ai.save();
        }
        applyTimeScale(g_ai.active);
        this->onResume(nullptr);
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
        }
    }

    void levelComplete() {
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
            f->label->setString(fmt::format("AI attempt {}  best {:.1f}%", g_ai.attempts, std::min(pct, 100.f)).c_str());
        }
    }
};

// ---------- the control loop ----------

class $modify(AIGameLayer, GJBaseGameLayer) {
    void update(float dt) {
        GJBaseGameLayer::update(dt);
        auto pl = PlayLayer::get();
        if (!g_ai.active || !pl || static_cast<GJBaseGameLayer*>(pl) != this) return;
        static_cast<AIPlayLayer*>(pl)->updateAILabel();
        if (g_ai.episodeOver || m_player1->m_isDead) return;

        auto p = m_player1;
        float x = p->getPositionX();
        g_ai.bestX = std::max(g_ai.bestX, x);

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
