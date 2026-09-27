// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "surface_types.h"

namespace neurale::experiment_presentation
{

[[nodiscard]] bool valid_utf8(std::string_view text) noexcept;

using PreparedTextId = std::uint64_t;

struct TextAtlasConfig
{
    std::string_view font_path{};
    long face_idx{};
    unsigned int pixel_height{};
    int atlas_width{};
    int atlas_height{};
    std::size_t max_texts{};
    std::size_t max_glyphs{};
    /// When non-empty, the exact font bytes FreeType consumes via
    /// FT_New_Memory_Face, so the rasterized/shaped font is the same buffer the
    /// presenter hashed for FontResourceIdentity. Empty falls back to opening
    /// font_path with FT_New_Face (used by atlas-only tests that do not record
    /// provenance and therefore cannot suffer the second-read divergence).
    std::span<const std::byte> font_bytes{};
};

struct TextCatalogEntry
{
    PreparedTextId id{};
    std::string_view utf8{};
};

struct GlyphAtlasEntry
{
    std::uint32_t glyph_id{};
    int atlas_x{};
    int atlas_y{};
    int width{};
    int height{};
    int bitmap_left{};
    int bitmap_top{};
};

struct ShapedGlyph
{
    std::size_t atlas_entry{};
    float x_offset{};
    float y_offset{};
    float x_advance{};
    float y_advance{};
};

struct PreparedText
{
    PreparedTextId id{};
    std::string utf8{};
    std::vector<ShapedGlyph> glyphs{};
};

struct TextPreparationStats
{
    std::uint64_t total_ns{};
    std::uint64_t shaping_ns{};
    std::uint64_t glyph_cache_ns{};
};

class PreparedTextAtlas
{
  public:
    [[nodiscard]] SurfaceStatus prepare(const TextAtlasConfig& config,
                                        std::span<const TextCatalogEntry> catalog);
    void close() noexcept;

    [[nodiscard]] const PreparedText* find(PreparedTextId id) const noexcept;
    [[nodiscard]] const GlyphAtlasEntry& glyph(std::size_t idx) const noexcept
    {
        return glyphs_[idx];
    }
    [[nodiscard]] std::span<const std::uint8_t> pixels() const noexcept
    {
        return pixels_;
    }
    [[nodiscard]] int width() const noexcept
    {
        return width_;
    }
    [[nodiscard]] int height() const noexcept
    {
        return height_;
    }
    [[nodiscard]] std::size_t text_count() const noexcept
    {
        return texts_.size();
    }
    [[nodiscard]] std::size_t glyph_count() const noexcept
    {
        return glyphs_.size();
    }
    [[nodiscard]] bool prepared() const noexcept
    {
        return prepared_;
    }
    [[nodiscard]] TextPreparationStats preparation_stats() const noexcept
    {
        return preparation_stats_;
    }

  private:
    std::vector<std::uint8_t> pixels_{};
    std::vector<GlyphAtlasEntry> glyphs_{};
    std::vector<PreparedText> texts_{};
    int width_{};
    int height_{};
    TextPreparationStats preparation_stats_{};
    bool prepared_{};
};

} // namespace neurale::experiment_presentation
