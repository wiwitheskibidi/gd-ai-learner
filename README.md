# GD AI Learner (Android)

A Geometry Dash launcher (forked from the open-source Geode launcher) that
comes with only one mod: **AI Learner**. In any level, pause and press
**AI Learn**: a reinforcement-learning agent takes over and gets further each
attempt. Pause and press **Stop AI** to play yourself.

It uses your own Geometry Dash install (Play Store, version 2.2081). The game
itself is not included or modified.

## Get the APK
Newest build (no sign-in needed):
https://github.com/wiwitheskibidi/gd-ai-learner/releases/latest/download/gd-ai-learner.apk

Every push to `main` builds the APK on GitHub Actions (~10 min) and publishes it as a
release named `build-<number>`; older builds stay on the Releases page.

## Install on BlueStacks
Drag `gd-ai-learner.apk` onto the BlueStacks window (or use "Install APK"
in the side bar). Open **GD AI Learner**; the first launch downloads the Geode
loader, then press Play. Use a 64-bit BlueStacks instance (Pie64 / Android 11).

It installs next to normal Geode with its own separate mods folder and settings.

## Layout
- `mod/`: the AI mod (C++, Geode SDK). Logic is in `mod/src/main.cpp`.
- `app/`: the Android launcher. Changes from upstream: app name, package id
  `com.william.gdailearner`, the bundled mod is installed on every launch
  (`installBundledMods` in `GeometryDashActivity.kt`), and launcher self-updates
  are disabled so it never gets replaced by stock Geode.

Licensed under the upstream launcher's license (LICENSE.txt).
Don't submit AI completions as your own.
