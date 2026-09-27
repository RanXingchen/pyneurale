/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "fitted_feature_contract.h"

namespace neurale::pipeline
{

void apply_feature_scaling(FeatureScaling scaling, std::span<double> row,
                           std::span<const double> center, std::span<const double> scale) noexcept
{
    switch (scaling)
    {
    case FeatureScaling::none:
        return;
    case FeatureScaling::standard:
        for (std::size_t i = 0; i < row.size(); ++i)
        {
            row[i] = (row[i] - center[i]) / scale[i];
        }
        return;
    case FeatureScaling::minmax:
        for (std::size_t i = 0; i < row.size(); ++i)
        {
            row[i] = row[i] * scale[i] + center[i];
        }
        return;
    }
}

} // namespace neurale::pipeline
