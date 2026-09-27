// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "surface_types.h"
#include "text_atlas.h"

namespace neurale::experiment_presentation
{

struct FontResourceIdentity
{
    std::string path{};
    long face_idx{};
    std::array<std::uint8_t, 32> sha256{};
};

/// The font bytes read exactly once, plus the identity derived from those same
/// bytes. The presenter hashes these bytes for provenance and feeds them to
/// FreeType via FT_New_Memory_Face, so the recorded SHA-256 is always the
/// SHA-256 of the bytes actually rasterized/shaped -- the path is never opened a
/// second time between hashing and glyph preparation. The bytes need not outlive
/// preparation: once the atlas is built and the FT_Face is done, they are released.
struct PreparedFontResource
{
    std::vector<std::byte> bytes{};
    FontResourceIdentity identity{};
};

/// Read the font file selected by config.font_path exactly once, producing the
/// PreparedFontResource above.
[[nodiscard]] SurfaceStatus load_font_resource(const TextAtlasConfig& config,
                                               PreparedFontResource& resource);

} // namespace neurale::experiment_presentation
