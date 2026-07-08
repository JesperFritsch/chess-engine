#!/usr/bin/env bash
#
# Launch the chess TUI in its own Ghostty window with a larger font and a clean
# chess-piece fallback font, without changing your normal terminal config.
#
# - The primary font stays JetBrainsMono Nerd Font so board labels/text keep
#   their monospace alignment.
# - Noto Sans Symbols 2 is added as a fallback so the piece glyphs
#   (U+265A..U+265F) render from that font instead of emoji.
# - gtk-single-instance=false forces a fresh instance, so these per-window font
#   overrides actually take effect (the default "detect" can route the window
#   through an existing instance and ignore them).
#
# Tweak FONT_SIZE / the fallback font below to taste.

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FONT_SIZE=16

# Build the optimized binary (fast no-op if already up to date). This runs in
# the launching terminal; the game opens in a new window.
cargo build --release --manifest-path "$DIR/Cargo.toml"

exec ghostty \
    --font-family="JetBrainsMono Nerd Font" \
    --font-family="Noto Sans Symbols 2" \
    --font-size="$FONT_SIZE" \
    --gtk-single-instance=false \
    -e "$DIR/target/release/chess-engine"
