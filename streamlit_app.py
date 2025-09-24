import pandas as pd
import streamlit as st
import numpy as np
from pathlib import Path
import base64
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle  # figür çerçevesi


st.set_page_config(page_title="Perfume Dashboard",
                   page_icon="images/perfume.png",
                   layout="wide")

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"]   = "white"


FIG_W, FIG_H   = 8.6, 4.5
FIG_W2, FIG_H2 = 8.8, 3.6
TL_PAD, TL_WPAD, TL_HPAD = 0.12, 0.15, 0.12

@st.cache_data
def get_data():
    return pd.read_csv("Data/Perfumes_dataset_clean.csv")

df = get_data()

st.markdown("""
<style>
div.block-container{ padding-top: 4rem; padding-bottom:.6rem; }

div[data-testid="stVerticalBlock"]{ gap:.08rem !important; }
div[data-testid="stHorizontalBlock"]{ gap:.12rem !important; }

div[data-testid="column"]{
  padding-left:.10rem !important;
  padding-right:.10rem !important;
}

div.element-container{ margin-bottom:.10rem !important; }

div[data-testid="stImage"]{ margin:0 !important; }
div[data-testid="stImage"] img{ display:block; margin:0 !important; }

.chart-card{ background:transparent; border:none; padding:0; margin:0 0 1px 0; }
.chart-inner{
  background:#ffffff; border:none; border-radius:8px; padding:3px;
  box-shadow:0 1px 6px rgba(0,0,0,.06);
}

.header-wrap{ padding-bottom:2rem; }
.page-title{
  text-align:center; font-size:32px; font-weight:800;
  margin:.35rem 0 .15rem 0; line-height:1.2; color:#0f172a;
}
.page-subtitle{
  text-align:center; color:#667085; font-style:italic;
  margin:0; line-height:1.25;
}
</style>
""", unsafe_allow_html=True)

def chart_card(fig, frame_color="#d0d5dd", lw=1.0, pad=0.002):
    fig.patch.set_facecolor("white")
    for ax in fig.get_axes():
        ax.set_facecolor("white")

    fig.add_artist(
        Rectangle((pad, pad), 1-2*pad, 1-2*pad,
                  fill=False, transform=fig.transFigure,
                  linewidth=lw, edgecolor=frame_color)
    )

    st.markdown('<div class="chart-card"><div class="chart-inner">', unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div></div>', unsafe_allow_html=True)
    plt.close(fig)

for col in df.columns:
    if pd.api.types.is_object_dtype(df[col]):
        df[col] = df[col].astype(str).str.strip()

brand_col    = "brand" if "brand" in df.columns else None
category_col = "category" if "category" in df.columns else None
aud_col      = "target_audience"

st.markdown(
    """
    <div class="header-wrap">
        <div class="page-title">PERFUME MARKET INSIGHTS</div>
        <div class="page-subtitle">
            where data meets fragrance: insights you can almost smell
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

def img_to_base64(path: str) -> str:
    p = Path(path)
    return base64.b64encode(p.read_bytes()).decode("utf-8")

with st.sidebar:
    try:
        b64 = img_to_base64("images/filter.png")
        st.markdown(
            f"""
            <div style="display:flex; align-items:center;">
                <img src="data:image/png;base64,{b64}" width="18" style="margin-right:8px;" alt="icon"/>
                <span style="font-size:20px; font-weight:bold;">Filters</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception:
        st.markdown("### Filters")

    if brand_col:
        brand_options = sorted(df[brand_col].dropna().unique().tolist())
        sel_brand = st.multiselect("Brand", options=brand_options,
                                   default=brand_options, key="f_brand")
    else:
        sel_brand, brand_options = [], []

    if category_col:
        category_options = sorted(df[category_col].dropna().unique().tolist())
        sel_cat = st.multiselect("Category", options=category_options,
                                 default=category_options, key="f_category")
    else:
        sel_cat, category_options = [], []

    aud_options = sorted(df[aud_col].dropna().unique().tolist())
    sel_aud = st.multiselect("Target Audience", options=aud_options,
                             default=aud_options, key="f_audience")

    if st.button("Clear filters"):
        st.session_state["f_brand"] = brand_options
        st.session_state["f_category"] = category_options
        st.session_state["f_audience"] = aud_options
        st.rerun()

fdf = df.copy()
if brand_col and sel_brand:
    fdf = fdf[fdf[brand_col].isin(sel_brand)]
if category_col and sel_cat:
    fdf = fdf[fdf[category_col].isin(sel_cat)]
if sel_aud:
    fdf = fdf[fdf[aud_col].isin(sel_aud)]

top_left, top_right = st.columns([3, 2], gap="small")

with top_left:
    if fdf.empty or (brand_col is None) or (category_col is None):
        st.info("No data for current filter selection or required columns are missing.")
    else:
        brand_counts = fdf[brand_col].value_counts().nlargest(10)
        brand_counts_sorted = brand_counts.sort_values()

        brand_div = (
            fdf.groupby(brand_col)[category_col]
               .nunique()
               .reindex(brand_counts.index)
        )
        brand_div_sorted = brand_div.sort_values()

        max1 = int(brand_counts_sorted.max() or 0)
        pad1 = max(2, int(max1 * 0.06))

        fig, axes = plt.subplots(1, 2, figsize=(FIG_W2, FIG_H2))

        brand_counts_sorted.plot(kind="barh", ax=axes[0], color="#219EBC")
        axes[0].set_title("Top 10 Brands by Number of Perfumes", fontsize=8)
        axes[0].set_xlabel("Total Perfumes", fontsize=7)
        axes[0].set_ylabel("Brand", fontsize=7)
        for i, v in enumerate(brand_counts_sorted.values):
            axes[0].text(v + 0.5, i, str(int(v)), va="center", fontsize=6)
        axes[0].tick_params(axis='y', labelsize=6)
        axes[0].tick_params(axis='x', labelsize=6)
        axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)
        axes[0].set_xlim(0, max1 + pad1)

        brand_div_sorted.plot(kind="barh", ax=axes[1], color="#126782")
        axes[1].set_title("Category Diversity of Top 10 Brands", fontsize=8)
        axes[1].set_xlabel("Unique Categories", fontsize=7)
        axes[1].set_ylabel("Brand", fontsize=7)
        for i, v in enumerate(brand_div_sorted.values):
            axes[1].text(v + 0.5, i, str(int(v)), va="center", fontsize=6)
        axes[1].tick_params(axis='y', labelsize=6)
        axes[1].tick_params(axis='x', labelsize=6)
        axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)
        axes[1].set_xlim(0, 80)

        plt.tight_layout(pad=TL_PAD, w_pad=TL_WPAD, h_pad=TL_HPAD)
        chart_card(fig)

with top_right:
    c1, c2 = st.columns(2, gap="small")
    with c1:
        st.markdown(
            f"""
            <div style="background-color:#f5f5f5;padding:8px;border-radius:8px;text-align:center;min-height:64px;">
                <div style="font-size:13px;color:#555;font-weight:600;">Total Brand</div>
                <div style="font-size:18px;font-weight:700;color:#000;margin-top:2px;">
                    {fdf[brand_col].nunique() if brand_col else 0}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""
            <div style="background-color:#f5f5f5;padding:8px;border-radius:8px;text-align:center;min-height:64px;">
                <div style="font-size:13px;color:#555;font-weight:600;">Total Perfume</div>
                <div style="font-size:18px;font-weight:700;color:#000;margin-top:2px;">
                    {fdf['perfume'].nunique() if 'perfume' in fdf.columns else fdf.shape[0]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    d = fdf.copy()
    if d.empty or not set(["category", "target_audience"]).issubset(d.columns):
        st.info("No data for current filter selection (or required columns missing).")
    else:
        d = d[d["category"].astype(str).str.lower() != "unknown"]
        col_order = [c for c in ["Female", "Male", "Unisex"] if c in d["target_audience"].unique()]
        ct = pd.crosstab(d["category"], d["target_audience"]).reindex(columns=col_order)
        top_categories = d["category"].value_counts().nlargest(15).index
        ct_top = ct.loc[top_categories].fillna(0).astype(int)

        fig5, ax5 = plt.subplots(figsize=(FIG_W, FIG_H))
        im = ax5.imshow(ct_top.values, cmap="Oranges", aspect="auto")
        ax5.set_yticks(np.arange(len(ct_top.index))); ax5.set_yticklabels(ct_top.index, fontsize=6)
        ax5.set_xticks(np.arange(len(ct_top.columns))); ax5.set_xticklabels(ct_top.columns, fontsize=6)
        ax5.set_title("Top Fragrance Categories by Target Audience", fontsize=8, pad=6)
        ax5.set_ylabel("Fragrance Category", fontsize=7); ax5.set_xlabel("", fontsize=7)
        for i in range(ct_top.shape[0]):
            for j in range(ct_top.shape[1]):
                ax5.text(j, i, str(ct_top.iat[i, j]), ha="center", va="center", fontsize=6, color="#222")
        cbar = fig5.colorbar(im, ax=ax5); cbar.ax.tick_params(labelsize=6)
        ax5.set_xticks(np.arange(-.5, ct_top.shape[1], 1), minor=True)
        ax5.set_yticks(np.arange(-.5, ct_top.shape[0], 1), minor=True)
        ax5.grid(which="minor", color="white", linestyle='-', linewidth=.5)
        ax5.tick_params(which="minor", bottom=False, left=False)
        ax5.spines["top"].set_visible(False); ax5.spines["right"].set_visible(False)
        plt.tight_layout(pad=TL_PAD, w_pad=TL_WPAD, h_pad=TL_HPAD)
        chart_card(fig5)

r2_left, r2_right = st.columns([1, 1], gap="small")

with r2_left:
    d = fdf
    if d.empty or not set(["type", "target_audience"]).issubset(d.columns):
        st.info("No data for current filter selection (or required columns missing).")
    else:
        N = 8
        top_types = d["type"].value_counts().nlargest(N).index
        tmp = d[d["type"].isin(top_types)].copy()

        ct = pd.crosstab(tmp["type"], tmp["target_audience"])
        order = ["Female", "Male", "Unisex"]
        ct = ct.reindex(columns=[c for c in order if c in ct.columns])

        pct = ct.div(ct.sum(axis=1), axis=0) * 100
        pct = pct.loc[pct.sum(axis=1).sort_values(ascending=False).index]

        color_map = {"Female": "#9B2226", "Male": "#023047", "Unisex": "#FD9E02"}
        colors = [color_map[c] for c in pct.columns]

        fig3, ax3 = plt.subplots(figsize=(FIG_W, FIG_H))
        pct.plot(kind="bar", stacked=True, color=colors, ax=ax3)

        ax3.set_title("Perfume Concentration Type by Target Audience (%)", fontsize=8, pad=6)
        ax3.set_xlabel("Concentration Type", fontsize=7)
        ax3.set_ylabel("", fontsize=7)
        ax3.set_ylim(0, 100)
        ax3.tick_params(axis="x", labelsize=6, rotation=20)
        ax3.tick_params(axis="y", labelsize=6)
        ax3.legend(title="Target Audience", bbox_to_anchor=(1.02, 1), loc="upper left",
                   fontsize=6, title_fontsize=6)

        for container in ax3.containers:
            labels = [f"{v:.1f}%" if (0 < v < 100) else "" for v in container.datavalues]
            ax3.bar_label(container, labels=labels, label_type="center",
                          fontsize=6, color="white", fontweight="bold")

        ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)
        plt.tight_layout(pad=TL_PAD, w_pad=TL_WPAD, h_pad=TL_HPAD)
        chart_card(fig3)

with r2_right:
    d = fdf.copy()
    need = {"category", "longevity"}
    if d.empty or not need.issubset(d.columns):
        st.info("No data for current filter selection (or required columns missing).")
    else:
        TOP_N = 10
        lon_order = ["Light","Light-Medium","Medium","Medium-Strong","Strong","Very Strong"]
        colors = {"Light":"#FFB703","Light-Medium":"#FBBF24","Medium":"#F59E0B",
                  "Medium-Strong":"#D97706","Strong":"#B91C1C","Very Strong":"#7F1D1D"}

        d = d[d["category"].astype(str).str.lower() != "unknown"]
        top_cats = d["category"].value_counts().nlargest(TOP_N).index

        ct_counts = pd.crosstab(d["category"], d["longevity"]).loc[top_cats]
        ct_counts = ct_counts.reindex(columns=[c for c in lon_order if c in ct_counts.columns], fill_value=0)

        order_idx = (ct_counts["Strong"].sort_values(ascending=True).index
                     if "Strong" in ct_counts.columns else
                     ct_counts.sum(1).sort_values().index)
        ct_counts = ct_counts.loc[order_idx]

        ct_pct = (ct_counts.div(ct_counts.sum(axis=1), axis=0) * 100).round(1)

        fig6, ax6 = plt.subplots(figsize=(FIG_W, FIG_H))
        left_vals = np.zeros(len(ct_pct)); y = np.arange(len(ct_pct.index))
        for col in ct_pct.columns:
            ax6.barh(y, ct_pct[col].values, left=left_vals, height=0.7,
                     label=col, color=colors.get(col, "#999999"),
                     edgecolor="white", linewidth=0.5)
            left_vals += ct_pct[col].values

        ax6.set_yticks(y); ax6.set_yticklabels(ct_pct.index, fontsize=6)
        ax6.set_xlim(0, 100)
        ax6.set_xlabel("Share within Category(%)", fontsize=8)
        ax6.set_ylabel("Fragrance Category", fontsize=8)
        ax6.set_title("Longevity Distribution across Top Fragrance Categories", fontsize=8, pad=6)
        ax6.tick_params(axis='x', labelsize=6)

        for container in ax6.containers:
            for rect, v in zip(container, container.datavalues):
                if 10 <= v < 100:
                    ax6.text(rect.get_x() + rect.get_width()/2,
                             rect.get_y() + rect.get_height()/2,
                             f"{v:.0f}%", ha="center", va="center",
                             fontsize=6, fontweight="bold", color="white")

        ax6.grid(axis="x", linestyle=":", alpha=0.35)
        ax6.spines["top"].set_visible(False); ax6.spines["right"].set_visible(False)
        ax6.legend(title="Longevity", loc="upper left", bbox_to_anchor=(1.02, 1),
                   frameon=True, fontsize=6, title_fontsize=6)

        plt.tight_layout(pad=TL_PAD, w_pad=TL_WPAD, h_pad=TL_HPAD)
        chart_card(fig6)

with st.container():
    r3_left, r3_right = st.columns([1, 1], gap="small")

    with r3_left:
        d = fdf
        required = {"type", "longevity"}
        if d.empty or not required.issubset(d.columns):
            st.info("No data for current filter selection (or required columns missing).")
        else:
            lon_order = ["Light","Light-Medium","Medium","Medium-Strong","Strong","Very Strong"]
            colors = {"Light":"#FFB703","Light-Medium":"#FBBF24","Medium":"#F59E0B",
                      "Medium-Strong":"#D97706","Strong":"#B91C1C","Very Strong":"#7F1D1D"}

            ct_counts = pd.crosstab(d["type"], d["longevity"])
            ct_counts = ct_counts.reindex(columns=[c for c in lon_order if c in ct_counts.columns], fill_value=0)

            row_totals = ct_counts.sum(axis=1)
            ct_counts = ct_counts.loc[row_totals[row_totals >= 10].index]
            order_by_total = ct_counts.sum(axis=1).sort_values(ascending=True).index
            ct_counts = ct_counts.loc[order_by_total]

            ct_pct = (ct_counts.div(ct_counts.sum(axis=1), axis=0) * 100).round(1)

            fig4, ax4 = plt.subplots(figsize=(FIG_W, FIG_H))
            left_vals = np.zeros(len(ct_pct)); y = np.arange(len(ct_pct.index))
            for col in ct_pct.columns:
                ax4.barh(y, ct_pct[col].values, left=left_vals, height=0.7,
                         label=col, color=colors.get(col, "#999999"),
                         edgecolor="white", linewidth=0.5)
                left_vals += ct_pct[col].values

            ax4.set_yticks(y); ax4.set_yticklabels(ct_pct.index, fontsize=6)
            ax4.set_xlabel("Share within Type (%)", fontsize=8)
            ax4.set_ylabel("Concentration Type", fontsize=8)
            ax4.set_title("Longevity Distribution", pad=6, fontsize=8)
            ax4.tick_params(axis='x', labelsize=6)

            xmax = 100; ax4.set_xlim(0, xmax + 10)
            for i, t in enumerate(ct_counts.index):
                ax4.text(
                    xmax + 2, i, f"n={int(ct_counts.loc[t].sum())}",
                    ha="left", va="center", fontsize=7, color="#444"
                )

            for container in ax4.containers:
                for rect, v in zip(container, container.datavalues):
                    if 8 <= v < 100:
                        ax4.text(rect.get_x() + rect.get_width()/2,
                                 rect.get_y() + rect.get_height()/2,
                                 f"{v:.1f}%", ha="center", va="center",
                                 fontsize=6, fontweight="bold", color="white")

            ax4.legend(title="Longevity", bbox_to_anchor=(1.02,1), loc="upper left",
                       frameon=False, fontsize=6, title_fontsize=6)
            ax4.spines['top'].set_visible(False); ax4.spines['right'].set_visible(False)

            plt.tight_layout(pad=TL_PAD, w_pad=TL_WPAD, h_pad=TL_HPAD)
            chart_card(fig4)

    with r3_right:
        d = fdf.copy()
        need = {"brand", "longevity"}
        if d.empty or not need.issubset(d.columns):
            st.info("No data for current filter selection (or required columns missing).")
        else:
            def longevity_group(val):
                if val in ["Light", "Light-Medium", "Light–Medium"]: return "Light"
                elif val == "Medium": return "Medium"
                elif val in ["Medium-Strong", "Strong", "Very Strong"]: return "StrongPlus"
                return "Other"

            d["lon_group"] = d["longevity"].apply(longevity_group)
            lon_ct  = pd.crosstab(d["brand"], d["lon_group"])
            lon_pct = lon_ct.div(lon_ct.sum(axis=1), axis=0) * 100

            brand_counts = d["brand"].value_counts().nlargest(10)
            top_brands   = brand_counts.index

            plot_df = pd.DataFrame({
                "Light %":   (lon_pct.loc[top_brands, "Light"] if "Light" in lon_pct.columns else 0) * -1,
                "Strong+ %": (lon_pct.loc[top_brands, "StrongPlus"] if "StrongPlus" in lon_pct.columns else 0),
                "Medium %":  (lon_pct.loc[top_brands, "Medium"] if "Medium" in lon_pct.columns else 0),
                "Total":      brand_counts.loc[top_brands]
            }).fillna(0).sort_values("Total", ascending=True)

            max_left  = abs(plot_df["Light %"].min())
            max_right = plot_df["Strong+ %"].max()
            xlim = max(max_left, max_right) + 5

            fig7, ax7 = plt.subplots(figsize=(FIG_W, FIG_H))
            color_light  = "#FFB703"
            color_strong = "#B91C1C"

            ax7.barh(plot_df.index, plot_df["Light %"],  color=color_light,  edgecolor="none",
                     label="Light / Light-Medium")
            ax7.barh(plot_df.index, plot_df["Strong+ %"], color=color_strong, edgecolor="none",
                     label="Medium-Strong / Strong / Very Strong")

            ax7.axvline(0, color="#F59E0B", linewidth=5)
            ax7.set_xlim(-xlim, xlim)
            ax7.set_xlabel("Share of Perfumes(%)", fontsize=8)
            ax7.set_ylabel("Brand", fontsize=8)
            ax7.set_title("Top 10 Brands by Longevity Strategy (Medium as Baseline)", fontsize=8, pad=6)
            ax7.tick_params(axis='x', labelsize=6); ax7.tick_params(axis='y', labelsize=6)

            threshold = 3
            for y_idx, (lt, stp) in enumerate(zip(plot_df["Light %"].values, plot_df["Strong+ %"].values)):
                if abs(lt) >= threshold:
                    ax7.text(lt/2, y_idx, f"{abs(lt):.0f}%", va="center", ha="center",
                             color="black", fontsize=6)
                if stp >= threshold:
                    ax7.text(stp/2, y_idx, f"{stp:.0f}%", va="center", ha="center",
                             color="white", fontsize=6, fontweight="bold")

            ax7.set_yticklabels([f"{b}  (n={int(plot_df.loc[b,'Total'])})" for b in plot_df.index], fontsize=6)
            ax7.legend(loc="upper left", bbox_to_anchor=(1.02, 1),
                       frameon=True, fontsize=6, title_fontsize=6)
            ax7.spines["top"].set_visible(False); ax7.spines["right"].set_visible(False)

            plt.tight_layout(pad=TL_PAD, w_pad=TL_WPAD, h_pad=TL_HPAD)
            chart_card(fig7)
