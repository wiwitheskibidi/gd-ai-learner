// AI Level Builder: an "AI Build" button in the editor's pause menu. You type a
// prompt and it fills the editor with a level. Uses Claude when an Anthropic API
// key is set in the mod settings, and an offline keyword-based generator otherwise.

#include <Geode/Geode.hpp>
#include <Geode/modify/EditorPauseLayer.hpp>
#include <Geode/ui/TextInput.hpp>
#include <Geode/utils/web.hpp>
#include <random>
#include <cctype>
#include <thread>
#include <algorithm>

using namespace geode::prelude;

namespace {

// One object on the editor grid: x/y in blocks (y = 0 sits on the ground).
struct GenObj {
    int id;
    int x;
    int y;
    bool flipped;
};

struct ObjType {
    char const* name;
    int id;
};

// Names Claude (and the offline generator) use, mapped to GD object IDs.
constexpr ObjType OBJECT_TYPES[] = {
    {"block", 1}, {"half_block", 40}, {"spike", 8}, {"small_spike", 39},
    {"yellow_pad", 35}, {"pink_pad", 140}, {"blue_pad", 67},
    {"yellow_orb", 36}, {"pink_orb", 141}, {"blue_orb", 84},
    {"cube_portal", 13}, {"ship_portal", 12}, {"ball_portal", 47}, {"ufo_portal", 111},
    {"wave_portal", 660}, {"robot_portal", 745}, {"spider_portal", 1331}, {"swing_portal", 1933},
    {"gravity_down_portal", 10}, {"gravity_up_portal", 11}, {"mini_portal", 101}, {"normal_size_portal", 99},
    {"speed_slow", 200}, {"speed_normal", 201}, {"speed_fast", 202}, {"speed_faster", 203}, {"speed_fastest", 1334},
};

int idFor(std::string_view name) {
    for (auto& t : OBJECT_TYPES) {
        if (name == t.name) return t.id;
    }
    return 0;
}

// ---------- offline generator ----------

bool has(std::string const& s, std::initializer_list<char const*> words) {
    for (auto w : words) {
        if (s.find(w) != std::string::npos) return true;
    }
    return false;
}

std::vector<GenObj> generateOffline(std::string prompt) {
    std::transform(prompt.begin(), prompt.end(), prompt.begin(), [](unsigned char c) { return std::tolower(c); });
    std::mt19937 rng(static_cast<uint32_t>(std::hash<std::string>{}(prompt)));
    auto roll = [&](int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); };

    int diff = 2;
    if (has(prompt, {"easy", "simple", "beginner"})) diff = 1;
    if (has(prompt, {"hard"})) diff = 3;
    if (has(prompt, {"insane", "extreme", "very hard"})) diff = 4;
    if (has(prompt, {"demon", "impossible", "nightmare"})) diff = 5;

    int length = 220;
    if (has(prompt, {"short", "tiny", "quick"})) length = 120;
    if (has(prompt, {"long"})) length = 400;
    if (has(prompt, {"very long", "xl", "huge"})) length = 650;
    // "<n> seconds": about 10.4 blocks per second at normal speed
    for (size_t i = 0; i < prompt.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(prompt[i]))) continue;
        size_t j = i;
        int n = 0;
        while (j < prompt.size() && std::isdigit(static_cast<unsigned char>(prompt[j]))) n = n * 10 + (prompt[j++] - '0');
        if (prompt.compare(j, 4, " sec") == 0 || prompt.compare(j, 3, "sec") == 0 || prompt.compare(j, 2, "s ") == 0) {
            length = std::clamp(static_cast<int>(n * 10.4f), 60, 1500);
        }
        i = j;
    }

    // game modes, in the order the prompt mentions them
    std::vector<std::pair<size_t, std::string>> found;
    for (auto mode : {"cube", "ship", "ball", "ufo", "wave", "robot", "spider", "swing"}) {
        auto pos = prompt.find(mode);
        if (pos != std::string::npos) found.push_back({pos, mode});
    }
    std::sort(found.begin(), found.end());
    std::vector<std::string> modes;
    for (auto& [pos, m] : found) modes.push_back(m);
    if (modes.empty()) modes = {"cube"};

    bool orbs = has(prompt, {"orb", "ring"});
    bool pads = has(prompt, {"pad"});

    std::vector<GenObj> out;
    int x = 8;   // lead-in so the player has room to start

    if (has(prompt, {"fastest"})) out.push_back({1334, 3, 1, false});
    else if (has(prompt, {"faster", "very fast"})) out.push_back({203, 3, 1, false});
    else if (has(prompt, {"fast"})) out.push_back({202, 3, 1, false});
    else if (has(prompt, {"slow"})) out.push_back({200, 3, 1, false});

    int segLen = std::max(40, (length - x) / static_cast<int>(modes.size()));
    for (size_t mi = 0; mi < modes.size(); ++mi) {
        auto const& mode = modes[mi];
        int segEnd = x + segLen;
        if (mi > 0 || mode != "cube") {
            int portal = idFor(mode + "_portal");
            if (portal) out.push_back({portal, x, 1, false});
            x += 4;
        }

        bool flying = mode == "ship" || mode == "ufo" || mode == "wave" || mode == "swing";
        if (flying) {
            int ceiling = std::max(5, 10 - diff);
            for (int bx = x; bx < segEnd; ++bx) out.push_back({1, bx, ceiling, false});
            int gap = std::max(2, 5 - diff * 6 / 10);
            int spacing = std::max(3, 9 - diff);
            for (int bx = x + 4; bx < segEnd - 2; bx += spacing + roll(0, 2)) {
                int gapStart = roll(1, std::max(1, ceiling - gap - 1));
                for (int by = 0; by < gapStart; ++by) out.push_back({1, bx, by, false});
                for (int by = gapStart + gap; by < ceiling; ++by) out.push_back({1, bx, by, false});
                if (diff >= 3) {
                    out.push_back({8, bx, gapStart, false});              // spike on the lower pillar
                    out.push_back({8, bx, gapStart + gap - 1, true});     // and under the upper one
                }
            }
        } else {
            bool roof = mode == "ball" || mode == "spider";
            if (roof) {
                for (int bx = x; bx < segEnd; ++bx) out.push_back({1, bx, 7, false});
            }
            int spacing = std::max(4, 10 - diff);
            for (int bx = x + 4; bx < segEnd - 4; bx += spacing + roll(0, 2)) {
                int pattern = roll(0, 9);
                if (orbs && pattern < 2) {
                    // spike pit with an orb above
                    for (int i = 0; i < 4; ++i) out.push_back({8, bx + i, 0, false});
                    out.push_back({36, bx + 1, 2, false});
                    bx += 3;
                } else if (pads && pattern < 4) {
                    out.push_back({35, bx, 0, false});
                    for (int i = 2; i < 6; ++i) out.push_back({8, bx + i, 0, false});
                    bx += 5;
                } else if (pattern < 7) {
                    int n = std::min(3, 1 + roll(0, std::min(diff, 3) - 1));
                    for (int i = 0; i < n; ++i) out.push_back({8, bx + i, 0, false});
                    bx += n;
                } else {
                    int w = roll(2, 3);
                    for (int i = 0; i < w; ++i) out.push_back({1, bx + i, 0, false});
                    if (diff >= 3) out.push_back({39, bx + w - 1, 1, false});
                    bx += w;
                }
                if (roof && diff >= 2 && roll(0, 2) == 0) out.push_back({8, bx + 2, 6, true});
            }
        }
        x = segEnd;
    }
    return out;
}

// ---------- Claude ----------

constexpr char const* SYSTEM_PROMPT = R"(You design Geometry Dash levels on a grid.
Coordinates are in blocks: x = 0 is the level start, x grows to the right; y = 0 is the row resting on the ground, y grows upward (max 20).
The player moves right at about 10.4 blocks per second at normal speed.
Physics to respect so the level is beatable:
- Cube: a jump reaches about 2 blocks high and clears at most 3 spikes in a row; leave at least 3 blocks between obstacles.
- Ship/UFO/wave/swing need a ceiling (a row of blocks) and openings at least 3 blocks tall (2 for wave on hard levels).
- Ball/spider flip gravity, so give them a ceiling row too.
- Put a portal before every game mode change and leave 6 empty blocks at the start.
- "flipped" turns an object upside down (use it for ceiling spikes).
Keep the level fun, readable and matched to the requested difficulty and length. Use at most 500 objects.)";

std::string schemaJson() {
    std::string names;
    for (auto& t : OBJECT_TYPES) {
        if (!names.empty()) names += ",";
        names += fmt::format("\"{}\"", t.name);
    }
    return fmt::format(R"({{"type":"object","additionalProperties":false,"required":["objects"],
"properties":{{"objects":{{"type":"array","items":{{"type":"object","additionalProperties":false,
"required":["type","x","y","flipped"],"properties":{{"type":{{"type":"string","enum":[{}]}},
"x":{{"type":"integer"}},"y":{{"type":"integer"}},"flipped":{{"type":"boolean"}}}}}}}}}}}})", names);
}

// Runs on a worker thread. Returns the objects, or an error message.
Result<std::vector<GenObj>> askClaude(std::string const& key, std::string const& model, std::string const& prompt) {
    auto schema = matjson::parse(schemaJson());
    if (!schema) return Err("internal schema error");

    auto body = matjson::makeObject({
        {"model", model},
        {"max_tokens", 32000},
        {"system", SYSTEM_PROMPT},
        {"output_config", matjson::makeObject({
            {"effort", "medium"},
            {"format", matjson::makeObject({{"type", "json_schema"}, {"schema", schema.unwrap()}})},
        })},
    });
    auto messages = matjson::Value::array();
    messages.push(matjson::makeObject({{"role", "user"}, {"content", "Build this level: " + prompt}}));
    body["messages"] = messages;

    geode::utils::web::WebRequest req;
    req.header("x-api-key", key);
    req.header("anthropic-version", "2023-06-01");
    req.header("content-type", "application/json");
    // Server-side fallback if the request is declined (models that support it).
    bool fallbackOk = model == "claude-opus-5-5" || model == "claude-opus-5" || model == "claude-sonnet-5-5" ||
                      model == "claude-fable-5-1";
    if (fallbackOk) {
        req.header("anthropic-beta", "server-side-fallback-2026-07-01");
        body["fallbacks"] = "default";
    }
    req.timeout(std::chrono::seconds(300));
    req.bodyJSON(body);

    auto res = req.postSync("https://api.anthropic.com/v1/messages");
    auto json = res.json();
    if (!json) return Err(fmt::format("network error (HTTP {})", res.code()));
    auto& j = json.unwrap();
    if (!res.ok()) {
        std::string msg = j["error"]["message"].asString().unwrapOr("unknown error");
        return Err(fmt::format("Claude API error {}: {}", res.code(), msg));
    }
    auto stop = j["stop_reason"].asString().unwrapOr("");
    if (stop == "refusal") return Err("Claude declined this prompt");
    if (stop == "max_tokens") return Err("level was too big; try a shorter one");

    std::string text;
    if (auto content = j["content"].asArray()) {
        for (auto& block : content.unwrap()) {
            if (block["type"].asString().unwrapOr("") == "text") text += block["text"].asString().unwrapOr("");
        }
    }
    auto level = matjson::parse(text);
    if (!level) return Err("Claude's answer could not be read");

    std::vector<GenObj> out;
    if (auto objs = level.unwrap()["objects"].asArray()) {
        for (auto& o : objs.unwrap()) {
            int id = idFor(o["type"].asString().unwrapOr(""));
            if (!id) continue;
            int x = static_cast<int>(o["x"].asInt().unwrapOr(-1));
            int y = static_cast<int>(o["y"].asInt().unwrapOr(-1));
            if (x < 0 || x > 5000 || y < 0 || y > 20) continue;
            out.push_back({id, x, y, o["flipped"].asBool().unwrapOr(false)});
        }
    }
    if (out.empty()) return Err("Claude returned an empty level");
    return Ok(out);
}

// ---------- placing objects in the editor ----------

int placeLevel(LevelEditorLayer* editor, std::vector<GenObj> const& objs, bool replace) {
    float baseX = 0.f;
    if (replace) {
        editor->removeAllObjects();
    } else if (editor->m_objects) {
        // append after whatever is already there
        for (auto obj : CCArrayExt<GameObject*>(editor->m_objects)) {
            if (obj) baseX = std::max(baseX, obj->getPositionX() + 120.f);
        }
    }
    std::string str;
    for (auto& o : objs) {
        str += fmt::format("1,{},2,{},3,{}", o.id, baseX + 15.f + 30.f * o.x, 15.f + 30.f * o.y);
        if (o.flipped) str += ",6,180";
        str += ";";
    }
    editor->createObjectsFromString(str, true, true);
    return static_cast<int>(objs.size());
}

}  // namespace

// ---------- the builder popup ----------

class AIBuildLayer : public CCLayerColor {
    EditorPauseLayer* m_pause = nullptr;
    TextInput* m_input = nullptr;
    CCLabelBMFont* m_status = nullptr;
    CCMenuItemSpriteExtra* m_replaceBtn = nullptr;
    bool m_replace = false;
    std::vector<CCNode*> m_hidden;

    static inline AIBuildLayer* s_open = nullptr;
    static inline bool s_busy = false;

    static ButtonSprite* small(char const* text, bool on) {
        return ButtonSprite::create(text, 120, true, "goldFont.fnt", on ? "GJ_button_02.png" : "GJ_button_04.png", 28.f, 0.6f);
    }

public:
    static AIBuildLayer* create(EditorPauseLayer* pause) {
        auto ret = new AIBuildLayer();
        if (ret->init(pause)) {
            ret->autorelease();
            return ret;
        }
        delete ret;
        return nullptr;
    }

    bool init(EditorPauseLayer* pause) {
        if (!CCLayerColor::initWithColor({0, 0, 0, 170})) return false;
        m_pause = pause;
        auto win = CCDirector::get()->getWinSize();
        CCPoint c = {win.width / 2, win.height / 2};

        for (auto child : CCArrayExt<CCNode*>(pause->getChildren())) {
            if (child->isVisible()) {
                child->setVisible(false);
                m_hidden.push_back(child);
            }
        }

        auto bg = CCScale9Sprite::create("GJ_square01.png");
        bg->setContentSize({400.f, 230.f});
        bg->setPosition(c);
        this->addChild(bg);

        auto title = CCLabelBMFont::create("AI Level Builder", "goldFont.fnt");
        title->setScale(0.8f);
        title->setPosition({c.x, c.y + 90.f});
        this->addChild(title);

        m_input = TextInput::create(360.f, "Describe a level, e.g. hard ship level with tight gaps", "chatFont.fnt");
        m_input->setCommonFilter(CommonFilter::Any);
        m_input->setMaxCharCount(300);
        m_input->setPosition({c.x, c.y + 45.f});
        this->addChild(m_input);

        bool claude = !Mod::get()->getSettingValue<std::string>("anthropic-api-key").empty();
        m_status = CCLabelBMFont::create(claude
            ? fmt::format("Using Claude ({})", Mod::get()->getSettingValue<std::string>("claude-model")).c_str()
            : "Offline generator (add an Anthropic API key in settings to use Claude)", "chatFont.fnt");
        m_status->setScale(0.55f);
        m_status->setPosition({c.x, c.y + 10.f});
        this->addChild(m_status);

        auto menu = CCMenu::create();
        menu->setPosition(c);
        this->addChild(menu);

        m_replaceBtn = CCMenuItemSpriteExtra::create(small("Add to level", false), this, menu_selector(AIBuildLayer::onReplace));
        m_replaceBtn->setPosition({-110.f, -30.f});
        menu->addChild(m_replaceBtn);

        auto gen = CCMenuItemSpriteExtra::create(
            ButtonSprite::create("Generate", 120, true, "goldFont.fnt", "GJ_button_01.png", 32.f, 0.8f),
            this, menu_selector(AIBuildLayer::onGenerate));
        gen->setPosition({50.f, -30.f});
        menu->addChild(gen);

        auto close = CCMenuItemSpriteExtra::create(
            ButtonSprite::create("Close", "goldFont.fnt", "GJ_button_04.png", 0.6f), this, menu_selector(AIBuildLayer::onClose));
        close->setPosition({0.f, -85.f});
        menu->addChild(close);

        s_open = this;
        return true;
    }

    void onExit() override {
        if (s_open == this) s_open = nullptr;
        CCLayerColor::onExit();
    }

    void setStatus(std::string const& text) { m_status->setString(text.c_str()); }

    void onReplace(CCObject*) {
        m_replace = !m_replace;
        m_replaceBtn->setNormalImage(small(m_replace ? "Replace level" : "Add to level", m_replace));
    }

    static void finish(std::vector<GenObj> const& objs, bool replace, std::string const& note) {
        s_busy = false;
        auto editor = LevelEditorLayer::get();
        if (!editor) return;
        int n = placeLevel(editor, objs, replace);
        auto msg = fmt::format("Placed {} objects{}", n, note);
        if (s_open) s_open->setStatus(msg);
        Notification::create(msg, NotificationIcon::Success)->show();
    }

    void onGenerate(CCObject*) {
        if (s_busy) return;
        std::string prompt = m_input->getString();
        if (prompt.empty()) {
            this->setStatus("Type what kind of level you want first");
            return;
        }
        bool replace = m_replace;
        auto key = Mod::get()->getSettingValue<std::string>("anthropic-api-key");
        if (key.empty()) {
            finish(generateOffline(prompt), replace, " (offline generator)");
            return;
        }
        auto model = Mod::get()->getSettingValue<std::string>("claude-model");
        s_busy = true;
        this->setStatus("Claude is designing your level... (this can take a minute)");
        std::thread([key, model, prompt, replace] {
            auto result = askClaude(key, model, prompt);
            geode::queueInMainThread([result = std::move(result), prompt, replace]() mutable {
                if (result) {
                    finish(result.unwrap(), replace, " (by Claude)");
                } else {
                    // fall back to the offline generator so the user still gets a level
                    finish(generateOffline(prompt), replace,
                        fmt::format(" (offline; Claude failed: {})", result.unwrapErr()));
                }
            });
        }).detach();
    }

    void onClose(CCObject*) {
        for (auto node : m_hidden) node->setVisible(true);
        this->removeFromParent();
    }
};

class $modify(AIEditorPauseLayer, EditorPauseLayer) {
    void customSetup() {
        EditorPauseLayer::customSetup();
        auto spr = ButtonSprite::create("AI Build", "goldFont.fnt", "GJ_button_01.png", 0.75f);
        auto btn = CCMenuItemSpriteExtra::create(spr, this, menu_selector(AIEditorPauseLayer::onAIBuild));
        btn->setID("ai-build-button"_spr);
        auto menu = CCMenu::create();
        menu->setID("ai-build-menu"_spr);
        menu->addChild(btn);
        auto win = CCDirector::get()->getWinSize();
        menu->setPosition({win.width / 2, 22.f});
        menu->setZOrder(100);
        this->addChild(menu);
    }

    void onAIBuild(CCObject*) {
        auto layer = AIBuildLayer::create(this);
        layer->setZOrder(1000);
        this->addChild(layer);
    }
};
