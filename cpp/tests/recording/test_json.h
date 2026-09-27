/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/// \file
/// A small read-only JSON reader, for test code only.
///
/// It exists so the C++ implementation can read `vectors/index.json` -- the
/// normative expectations of the recording-spool specification -- the same way any other
/// language would: by reading the committed file. The alternative, generating
/// a C++ header from the index with a Python script at build time, would make
/// the "language-neutral, needs no Python" claim in the specification false
/// for the one implementation that has to honour it.
///
/// It is not part of `neurale_recording`, and the container itself parses no
/// JSON anywhere: the plan document a spool stores is opaque bytes.

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace neurale::recording::test
{

class JsonValue
{
  public:
    enum class Type
    {
        null,
        boolean,
        integer,
        string,
        array,
        object,
    };

    Type type{Type::null};
    bool boolean{};
    std::int64_t integer{};
    std::string string{};
    std::vector<JsonValue> array{};
    std::vector<std::pair<std::string, JsonValue>> object{};

    [[nodiscard]] const JsonValue* find(std::string_view key) const noexcept
    {
        for (const auto& entry : object)
        {
            if (entry.first == key)
            {
                return &entry.second;
            }
        }
        return nullptr;
    }

    [[nodiscard]] bool is_null() const noexcept
    {
        return type == Type::null;
    }
};

class JsonParser
{
  public:
    explicit JsonParser(std::string_view text) noexcept : text_(text) {}

    [[nodiscard]] bool parse(JsonValue& out) noexcept
    {
        skip_space();
        if (!parse_value(out))
        {
            return false;
        }
        skip_space();
        return cursor_ == text_.size();
    }

  private:
    void skip_space() noexcept
    {
        while (cursor_ < text_.size() && (text_[cursor_] == ' ' || text_[cursor_] == '\t' ||
                                          text_[cursor_] == '\n' || text_[cursor_] == '\r'))
        {
            ++cursor_;
        }
    }

    [[nodiscard]] bool literal(std::string_view word) noexcept
    {
        if (text_.substr(cursor_, word.size()) != word)
        {
            return false;
        }
        cursor_ += word.size();
        return true;
    }

    [[nodiscard]] bool parse_string(std::string& out) noexcept
    {
        if (cursor_ >= text_.size() || text_[cursor_] != '"')
        {
            return false;
        }
        ++cursor_;
        out.clear();
        while (cursor_ < text_.size())
        {
            const auto character = text_[cursor_++];
            if (character == '"')
            {
                return true;
            }
            if (character != '\\')
            {
                out.push_back(character);
                continue;
            }
            if (cursor_ >= text_.size())
            {
                return false;
            }
            const auto escape = text_[cursor_++];
            switch (escape)
            {
            case '"':
                out.push_back('"');
                break;
            case '\\':
                out.push_back('\\');
                break;
            case '/':
                out.push_back('/');
                break;
            case 'b':
                out.push_back('\b');
                break;
            case 'f':
                out.push_back('\f');
                break;
            case 'n':
                out.push_back('\n');
                break;
            case 'r':
                out.push_back('\r');
                break;
            case 't':
                out.push_back('\t');
                break;
            case 'u':
            {
                if (cursor_ + 4 > text_.size())
                {
                    return false;
                }
                std::uint32_t code = 0;
                for (int i = 0; i < 4; ++i)
                {
                    const auto digit = text_[cursor_++];
                    code *= 16;
                    if (digit >= '0' && digit <= '9')
                    {
                        code += static_cast<std::uint32_t>(digit - '0');
                    }
                    else if (digit >= 'a' && digit <= 'f')
                    {
                        code += static_cast<std::uint32_t>(digit - 'a' + 10);
                    }
                    else if (digit >= 'A' && digit <= 'F')
                    {
                        code += static_cast<std::uint32_t>(digit - 'A' + 10);
                    }
                    else
                    {
                        return false;
                    }
                }
                // Basic multilingual plane only; the index has no surrogate
                // pairs and a test reader that pretended otherwise would be
                // untested code.
                if (code < 0x80)
                {
                    out.push_back(static_cast<char>(code));
                }
                else if (code < 0x800)
                {
                    out.push_back(static_cast<char>(0xC0 | (code >> 6)));
                    out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                }
                else
                {
                    out.push_back(static_cast<char>(0xE0 | (code >> 12)));
                    out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
                    out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                }
                break;
            }
            default:
                return false;
            }
        }
        return false;
    }

    [[nodiscard]] bool parse_value(JsonValue& out) noexcept
    {
        if (cursor_ >= text_.size())
        {
            return false;
        }
        const auto character = text_[cursor_];
        if (character == '{')
        {
            ++cursor_;
            out.type = JsonValue::Type::object;
            skip_space();
            if (cursor_ < text_.size() && text_[cursor_] == '}')
            {
                ++cursor_;
                return true;
            }
            while (true)
            {
                skip_space();
                std::string key;
                if (!parse_string(key))
                {
                    return false;
                }
                skip_space();
                if (cursor_ >= text_.size() || text_[cursor_] != ':')
                {
                    return false;
                }
                ++cursor_;
                skip_space();
                JsonValue value;
                if (!parse_value(value))
                {
                    return false;
                }
                out.object.emplace_back(std::move(key), std::move(value));
                skip_space();
                if (cursor_ < text_.size() && text_[cursor_] == ',')
                {
                    ++cursor_;
                    continue;
                }
                if (cursor_ < text_.size() && text_[cursor_] == '}')
                {
                    ++cursor_;
                    return true;
                }
                return false;
            }
        }
        if (character == '[')
        {
            ++cursor_;
            out.type = JsonValue::Type::array;
            skip_space();
            if (cursor_ < text_.size() && text_[cursor_] == ']')
            {
                ++cursor_;
                return true;
            }
            while (true)
            {
                skip_space();
                JsonValue value;
                if (!parse_value(value))
                {
                    return false;
                }
                out.array.push_back(std::move(value));
                skip_space();
                if (cursor_ < text_.size() && text_[cursor_] == ',')
                {
                    ++cursor_;
                    continue;
                }
                if (cursor_ < text_.size() && text_[cursor_] == ']')
                {
                    ++cursor_;
                    return true;
                }
                return false;
            }
        }
        if (character == '"')
        {
            out.type = JsonValue::Type::string;
            return parse_string(out.string);
        }
        if (literal("true"))
        {
            out.type = JsonValue::Type::boolean;
            out.boolean = true;
            return true;
        }
        if (literal("false"))
        {
            out.type = JsonValue::Type::boolean;
            out.boolean = false;
            return true;
        }
        if (literal("null"))
        {
            out.type = JsonValue::Type::null;
            return true;
        }

        const auto start = cursor_;
        if (cursor_ < text_.size() && (text_[cursor_] == '-' || text_[cursor_] == '+'))
        {
            ++cursor_;
        }
        while (cursor_ < text_.size() && text_[cursor_] >= '0' && text_[cursor_] <= '9')
        {
            ++cursor_;
        }
        if (cursor_ == start)
        {
            return false;
        }
        // The index carries integers only. A fractional or exponent form would
        // be a change to the index the reader should refuse rather than round.
        if (cursor_ < text_.size() &&
            (text_[cursor_] == '.' || text_[cursor_] == 'e' || text_[cursor_] == 'E'))
        {
            return false;
        }
        out.type = JsonValue::Type::integer;
        out.integer = std::stoll(std::string(text_.substr(start, cursor_ - start)));
        return true;
    }

    std::string_view text_;
    std::size_t cursor_{};
};

} // namespace neurale::recording::test
