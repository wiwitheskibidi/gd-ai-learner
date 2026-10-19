# GD AI Learner (Android)

A Geometry Dash launcher (forked from the open-source Geode launcher) that
comes with only one mod: **AI Learner**. In any level, pause and press
**AI Learn**: a reinforcement-learning agent takes over and gets further each
attempt. Pause and press **Stop AI** to play yourself.

It uses your own Geometry Dash install (Play Store, version 2.2081). The game
itself is not included or modified.

## Get the APK
1. Push this folder to a new GitHub repo.
2. GitHub > **Actions** > "Build GD AI Learner APK" runs automatically (~10 min).
3. Download the **gd-ai-learner-apk** artifact and unzip it.

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
