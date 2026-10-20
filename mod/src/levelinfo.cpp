// Adds an "AI Best" progress bar under the Normal/Practice bars on a level's page.

#include <Geode/Geode.hpp>
#include <Geode/modify/LevelInfoLayer.hpp>

using namespace geode::prelude;

class $modify(AILevelInfoLayer, LevelInfoLayer) {
    bool init(GJGameLevel* level, bool challenge) {
        if (!LevelInfoLayer::init(level, challenge)) return false;

        int id = level->m_levelID.value();
        float pct = Mod::get()->getSavedValue<float>(fmt::format("best-{}", id), -1.f);
        int attempts = Mod::get()->getSavedValue<int>(fmt::format("attempts-{}", id), 0);

        // Sit under the game's own "Practice Mode" bar when we can find it.
        auto win = CCDirector::get()->getWinSize();
        CCPoint labelPos = {win.width / 2, win.height / 2 - 70.f};
        for (auto child : CCArrayExt<CCNode*>(this->getChildren())) {
            if (auto lbl = typeinfo_cast<CCLabelBMFont*>(child)) {
                if (std::string_view(lbl->getString()) == "Practice Mode") {
                    labelPos = lbl->getPosition() - CCPoint{0.f, 36.f};
                }
            }
        }

        auto label = CCLabelBMFont::create("AI Best", "bigFont.fnt");
        label->setScale(0.5f);
        label->setPosition(labelPos);
        label->setID("ai-best-label"_spr);
        this->addChild(label);

        auto bar = CCSprite::create("GJ_progressBar_001.png");
        bar->setColor({0, 0, 0});
        bar->setOpacity(125);
        bar->setScale(0.6f);
        bar->setPosition(labelPos - CCPoint{0.f, 17.f});
        bar->setID("ai-best-bar"_spr);
        this->addChild(bar);

        auto size = bar->getContentSize();
        float shown = std::clamp(pct, 0.f, 100.f);
        if (shown > 0.f) {
            auto fill = CCSprite::create("GJ_progressBar_001.png");
            fill->setColor({40, 220, 255});
            fill->setAnchorPoint({0.f, 0.5f});
            fill->setTextureRect({0.f, 0.f, size.width * shown / 100.f, size.height});
            fill->setScaleX(0.992f);
            fill->setScaleY(0.86f);
            fill->setPosition({size.width * 0.004f, size.height / 2});
            bar->addChild(fill);
        }

        auto text = CCLabelBMFont::create(
            pct < 0.f ? "not trained yet" : fmt::format("{:.0f}%  ({} attempts)", shown, attempts).c_str(),
            "bigFont.fnt");
        text->setScale(0.35f);
        text->setPosition(bar->getPosition());
        this->addChild(text);
        return true;
    }
};
