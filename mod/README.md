# AI Learner (Geometry Dash mod)

Adds an **AI Learn** button to the pause menu. Press it on any level and a
reinforcement-learning agent (Q-learning) takes over: it plays, dies,
respawns and gets further each attempt. Press **Stop AI** (pause again) to
take back control. What it learns is saved per level, so it continues where it
left off next time.

## Build (no local setup needed)
1. Push this folder to a new GitHub repo.
2. GitHub > Actions > "Build Geode Mod" runs automatically.
3. Download the "Build Output" artifact: it contains `william.gd-ai-learner.geode`.

## Install on Android
1. Install the Geode launcher APK from https://geode-sdk.org (you need your own
   copy of Geometry Dash installed from the Play Store).
2. Copy the `.geode` file to Geode's mods folder
   (Geode > settings > Open mods folder), or install from file in the Geode menu.
3. Open a level, pause, press **AI Learn**.

## Notes
- Use practice-free normal mode; the agent restarts itself on each death.
- The "Training speed" setting speeds up the game while learning.
- Don't submit AI completions as your own (leaderboards / verification).
