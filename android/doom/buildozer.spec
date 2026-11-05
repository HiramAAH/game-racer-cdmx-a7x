[app]
title = GTC Racing
package.name = gtcracing
package.domain = org.hiram
source.dir = .
source.include_exts = py,png,jpg,wav,mp3,ogg
version = 1.0
requirements = python3,pygame,numpy
orientation = landscape
fullscreen = 1
android.archs = arm64-v8a, armeabi-v7a
android.api = 33
android.minapi = 21
android.accept_sdk_license = True
p4a.bootstrap = sdl2

[buildozer]
log_level = 2
warn_on_root = 0
