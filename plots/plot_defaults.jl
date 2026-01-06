module PlotDefaults
using Plots

const DOC_WIDTH_PT = 300
const DPI = 200
const GOLDEN_RATIO = (1 + sqrt(5)) / 2

pt_to_px(pt, dpi) = round(Int, pt / 72 * dpi)

function apply!(; doc_width_pt = DOC_WIDTH_PT, dpi = DPI, aspect = 1 / GOLDEN_RATIO,
                 guide_pt = 12, tick_pt = 12, legend_pt = 12)
    w = pt_to_px(doc_width_pt, dpi)
    h = round(Int, w * aspect)

    default(
        dpi = dpi,
        size = (w, h),
        guidefontsize = guide_pt,
        tickfontsize = tick_pt,
        legendfontsize = legend_pt,
        fontfamily = "Computer Modern",
    )
end

end
