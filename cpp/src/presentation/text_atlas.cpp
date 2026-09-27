// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT

#include "text_atlas.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <limits>

#include "utf8.h"

#include <ft2build.h>
#include FT_FREETYPE_H
#include <hb-ft.h>
#include <hb.h>

namespace neurale::experiment_presentation
{

bool valid_utf8(std::string_view text) noexcept
{
    // One decoder for the whole project: cpp/src/experiments/utf8.h is the
    // single implementation the catalog text contract trusts, the same one
    // the stimulus validator and the Python constructor call. A second
    // decoder here would be a second definition of "valid" that agrees today
    // and drifts later.
    return experiments::detail::is_well_formed_utf8(
        reinterpret_cast<const std::uint8_t*>(text.data()), text.size());
}

namespace
{

std::size_t find_glyph(std::span<const GlyphAtlasEntry> glyphs, std::uint32_t glyph_id) noexcept
{
    for (std::size_t i = 0; i < glyphs.size(); ++i)
    {
        if (glyphs[i].glyph_id == glyph_id)
        {
            return i;
        }
    }
    return glyphs.size();
}

} // namespace

SurfaceStatus PreparedTextAtlas::prepare(const TextAtlasConfig& config,
                                         std::span<const TextCatalogEntry> catalog)
{
    close();
    const auto total_begin = std::chrono::steady_clock::now();
    if (config.font_path.empty() || config.pixel_height == 0 || config.atlas_width <= 0 ||
        config.atlas_height <= 0 || config.max_texts == 0 || config.max_glyphs == 0 ||
        catalog.size() > config.max_texts ||
        static_cast<std::size_t>(config.atlas_width) >
            (std::numeric_limits<std::size_t>::max)() /
                static_cast<std::size_t>(config.atlas_height))
    {
        return SurfaceStatus::invalid_configuration;
    }

    FT_Library library{};
    if (FT_Init_FreeType(&library) != 0)
    {
        return SurfaceStatus::font_load_failed;
    }
    FT_Face face{};
    if (!config.font_bytes.empty())
    {
        // FT_New_Memory_Face keeps a pointer into font_bytes for the lifetime of
        // the face, which is destroyed at the end of this prepare() call before
        // the caller's buffer is released.
        if (FT_New_Memory_Face(library, reinterpret_cast<const FT_Byte*>(config.font_bytes.data()),
                               static_cast<FT_Long>(config.font_bytes.size()), config.face_idx,
                               &face) != 0)
        {
            FT_Done_FreeType(library);
            return SurfaceStatus::font_load_failed;
        }
    }
    else
    {
        const std::string font_path(config.font_path);
        if (FT_New_Face(library, font_path.c_str(), config.face_idx, &face) != 0)
        {
            FT_Done_FreeType(library);
            return SurfaceStatus::font_load_failed;
        }
    }
    if (FT_Set_Pixel_Sizes(face, 0, config.pixel_height) != 0)
    {
        FT_Done_Face(face);
        FT_Done_FreeType(library);
        return SurfaceStatus::font_load_failed;
    }
    hb_font_t* hb_font = hb_ft_font_create_referenced(face);
    if (hb_font == nullptr)
    {
        FT_Done_Face(face);
        FT_Done_FreeType(library);
        return SurfaceStatus::font_load_failed;
    }

    width_ = config.atlas_width;
    height_ = config.atlas_height;
    pixels_.assign(static_cast<std::size_t>(width_) * height_, 0);
    glyphs_.reserve(config.max_glyphs);
    texts_.reserve(config.max_texts);
    int pen_x = 1;
    int pen_y = 1;
    int row_height = 0;
    SurfaceStatus status = SurfaceStatus::ok;

    for (const auto& item : catalog)
    {
        if (!valid_utf8(item.utf8) || item.utf8.size() > INT_MAX)
        {
            status = SurfaceStatus::invalid_utf8;
            break;
        }
        if (find(item.id) != nullptr)
        {
            status = SurfaceStatus::invalid_configuration;
            break;
        }

        const auto shaping_begin = std::chrono::steady_clock::now();
        hb_buffer_t* buffer = hb_buffer_create();
        if (buffer == nullptr)
        {
            status = SurfaceStatus::resource_capacity_exceeded;
            break;
        }
        hb_buffer_add_utf8(buffer, item.utf8.data(), static_cast<int>(item.utf8.size()), 0,
                           static_cast<int>(item.utf8.size()));
        // Script and direction are derived from the unchanged Unicode text.
        // Freeze language to BCP-47 "und" instead of inheriting process locale.
        hb_buffer_set_language(buffer, hb_language_from_string("und", -1));
        hb_buffer_guess_segment_properties(buffer);
        hb_shape(hb_font, buffer, nullptr, 0);

        unsigned int n_glyphs{};
        const auto* infos = hb_buffer_get_glyph_infos(buffer, &n_glyphs);
        const auto* positions = hb_buffer_get_glyph_positions(buffer, &n_glyphs);
        const auto shaping_end = std::chrono::steady_clock::now();
        preparation_stats_.shaping_ns += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(shaping_end - shaping_begin)
                .count());
        // max_glyphs bounds both unique atlas glyphs (checked below when a new
        // glyph enters the atlas) and shaped glyphs per prepared text. A text
        // like "AAAAAAAA" has one unique glyph but eight shaped occurrences;
        // without this check it passes prepare (the single 'A' reuses one atlas
        // slot) yet exceeds the render VBO/batch capacity reserved from the same
        // max_glyphs, failing only at presentation time. Catch it at prepare.
        if (static_cast<std::size_t>(n_glyphs) > config.max_glyphs)
        {
            status = SurfaceStatus::resource_capacity_exceeded;
            hb_buffer_destroy(buffer);
            break;
        }
        PreparedText prepared{.id = item.id, .utf8 = std::string(item.utf8)};
        prepared.glyphs.reserve(n_glyphs);

        const auto glyph_begin = std::chrono::steady_clock::now();
        for (unsigned int shaped_idx = 0; shaped_idx < n_glyphs; ++shaped_idx)
        {
            const auto glyph_id = infos[shaped_idx].codepoint;
            if (glyph_id == 0)
            {
                status = SurfaceStatus::glyph_missing;
                break;
            }
            auto atlas_idx = find_glyph(glyphs_, glyph_id);
            if (atlas_idx == glyphs_.size())
            {
                if (glyphs_.size() == config.max_glyphs ||
                    FT_Load_Glyph(face, glyph_id, FT_LOAD_DEFAULT) != 0 ||
                    FT_Render_Glyph(face->glyph, FT_RENDER_MODE_NORMAL) != 0)
                {
                    status = glyphs_.size() == config.max_glyphs
                                 ? SurfaceStatus::atlas_capacity_exceeded
                                 : SurfaceStatus::glyph_missing;
                    break;
                }
                const auto bitmap_width = static_cast<int>(face->glyph->bitmap.width);
                const auto bitmap_height = static_cast<int>(face->glyph->bitmap.rows);
                if (bitmap_width > 0 && bitmap_height > 0)
                {
                    if (pen_x + bitmap_width + 1 > width_)
                    {
                        pen_x = 1;
                        pen_y += row_height + 1;
                        row_height = 0;
                    }
                    // A glyph wider than the atlas row does not become placeable
                    // by starting a new row, so the wrap above must be followed
                    // by a width check as well as the height check. Without it
                    // the row copy below runs past the end of its row -- silently
                    // corrupting the next glyph's pixels, and past the end of
                    // pixels_ entirely once pen_y is near the bottom. CJK
                    // stimuli make this reachable rather than theoretical: a
                    // Han glyph is roughly as wide as pixel_height, so any
                    // atlas_width below that is a configuration the caller can
                    // set (validate() only requires positive extents) and the
                    // honest answer is atlas_capacity_exceeded, not a truncated
                    // or overrunning blit.
                    if (pen_x + bitmap_width + 1 > width_ || pen_y + bitmap_height + 1 > height_)
                    {
                        status = SurfaceStatus::atlas_capacity_exceeded;
                        break;
                    }
                    const auto pitch = face->glyph->bitmap.pitch;
                    for (int row = 0; row < bitmap_height; ++row)
                    {
                        const auto source_row = pitch >= 0 ? row : bitmap_height - 1 - row;
                        const auto* source =
                            face->glyph->bitmap.buffer +
                            source_row * static_cast<std::size_t>(pitch >= 0 ? pitch : -pitch);
                        auto* destination =
                            pixels_.data() + static_cast<std::size_t>(pen_y + row) * width_ + pen_x;
                        std::copy_n(source, bitmap_width, destination);
                    }
                }
                glyphs_.push_back(GlyphAtlasEntry{
                    .glyph_id = glyph_id,
                    .atlas_x = pen_x,
                    .atlas_y = pen_y,
                    .width = bitmap_width,
                    .height = bitmap_height,
                    .bitmap_left = face->glyph->bitmap_left,
                    .bitmap_top = face->glyph->bitmap_top,
                });
                atlas_idx = glyphs_.size() - 1;
                if (bitmap_width > 0 && bitmap_height > 0)
                {
                    pen_x += bitmap_width + 1;
                    row_height = std::max(row_height, bitmap_height);
                }
            }
            prepared.glyphs.push_back(ShapedGlyph{
                .atlas_entry = atlas_idx,
                .x_offset = positions[shaped_idx].x_offset / 64.0F,
                .y_offset = positions[shaped_idx].y_offset / 64.0F,
                .x_advance = positions[shaped_idx].x_advance / 64.0F,
                .y_advance = positions[shaped_idx].y_advance / 64.0F,
            });
        }
        const auto glyph_end = std::chrono::steady_clock::now();
        preparation_stats_.glyph_cache_ns += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(glyph_end - glyph_begin).count());
        hb_buffer_destroy(buffer);
        if (status != SurfaceStatus::ok)
        {
            break;
        }
        texts_.push_back(std::move(prepared));
    }

    hb_font_destroy(hb_font);
    FT_Done_Face(face);
    FT_Done_FreeType(library);
    if (status != SurfaceStatus::ok)
    {
        close();
        return status;
    }
    prepared_ = true;
    preparation_stats_.total_ns =
        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now() - total_begin)
                                       .count());
    return SurfaceStatus::ok;
}

void PreparedTextAtlas::close() noexcept
{
    pixels_.clear();
    glyphs_.clear();
    texts_.clear();
    pixels_.shrink_to_fit();
    glyphs_.shrink_to_fit();
    texts_.shrink_to_fit();
    width_ = 0;
    height_ = 0;
    preparation_stats_ = {};
    prepared_ = false;
}

const PreparedText* PreparedTextAtlas::find(PreparedTextId id) const noexcept
{
    for (const auto& text : texts_)
    {
        if (text.id == id)
        {
            return &text;
        }
    }
    return nullptr;
}

} // namespace neurale::experiment_presentation
