#import dependendies 
import scipy
import pandas as pd
import numpy as np
import pynapple as nap
import matplotlib.pyplot as plt
import seaborn as sns
import requests, os
import xarray as xr 
from pathlib import Path 
import glob 
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import shapiro, f_oneway, kruskal
from statannotations.Annotator import Annotator
import scikit_posthocs as sp
import itertools


#import data

d_folder = "/Users/Nadia/data_analysis/ADN_Diestrus"
d_files = glob.glob(d_folder + "/**/*.nwb", recursive = True)
d_nwb = [nap.NWBFile(d) for d in d_files]

e_folder = "/Users/Nadia/data_analysis/ADN_Estrus"
e_files = glob.glob(e_folder + "/**/*.nwb", recursive = True)
e_nwb = [nap.NWBFile(e) for e in e_files]

m_folder = "/Users/Nadia/data_analysis/ADN_Mestrus"
m_files = glob.glob(m_folder + "/**/*.nwb", recursive = True)
m_nwb = [nap.NWBFile(m) for m in m_files]


p_folder = "/Users/Nadia/data_analysis/ADN_Proestrus"
p_files = glob.glob(p_folder + "/**/*.nwb", recursive = True)
p_nwb = [nap.NWBFile(p) for p in p_files]

md_nwb = d_nwb + m_nwb



def tuning_curves(NWB):
    
    #Create head direction tuning curves for each animal/stage (need spike timings and orientation of animal)
    spikes = NWB["units"]  # Get spike timings
    epochs = NWB["epochs"]  # Get the behavioural epochs (in this case, sleep1, exploration and sleep2)
    angle = NWB["ry"]  # Get the tracked orientation of the animal
        
    tuning_curves = nap.compute_tuning_curves(
        data=spikes, 
        features=angle, 
        bins=61, 
        epochs=epochs[epochs.tags == "exploration"], #sleep1, exploration, sleep2
        feature_names=["head_direction"]
        )

    MI = nap.compute_mutual_information(tuning_curves)
        
    mi_threshold = 0.4
        
    HD_cells = MI.loc[MI["bits/spike"]> mi_threshold].index 
    
    #Skip files with no HD cells
    if len(HD_cells) == 0:
        return None
        
    tuning_curves = tuning_curves.sel(unit=HD_cells)
            
    pref_ang = tuning_curves.idxmax(dim="head_direction")
  
     # Smooth curves
    tuning_curves.values = scipy.ndimage.gaussian_filter1d(
        tuning_curves.values, sigma=2, axis=1, mode="wrap"
    )
    return tuning_curves

def plot_curves(
    curves,
    stage,
    color,
    polar=False
):

    for tuning_curves in curves:

        # -----------------------------
        # Sort by preferred direction
        # -----------------------------
        pref_ang = tuning_curves.idxmax(dim="head_direction")
        sorted_curves = tuning_curves.sortby(pref_ang)

        n_units = len(sorted_curves.unit)
        if n_units == 0:
            print("No HD Cells")
            continue

        n_cols = 4
        n_rows = int(np.ceil(n_units / n_cols))

        # -----------------------------
        # Create subplots
        # -----------------------------
        if polar:
            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(4*n_cols, 3*n_rows),
                subplot_kw={'projection': 'polar'}
            )
        else:
            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(4*n_cols, 3*n_rows)
            )

        axes = axes.flatten()
        fig.subplots_adjust(wspace=0.4, hspace=0.6)

        theta = sorted_curves.coords["head_direction"].values

        # -----------------------------
        # Plot each unit
        # -----------------------------
        for i, unit in enumerate(sorted_curves.unit.values):
            ax = axes[i]
            r = sorted_curves.sel(unit=unit).values

            if polar:
                ax.plot(theta, r, color=color, linewidth=3)
                ax.fill(theta, r, color=color, alpha=0.3)
                ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4])
                ax.set_xticklabels(["0", "", "90","", "180","", "270", ""])
                ax.set_yticklabels([])

            else:
                ax.plot(theta, r, color=color, linewidth=3)
                ax.set_xlim(0, 2*np.pi)
                ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
                ax.set_xticklabels(["0", "90", "180", "270", "360"])
                ax.set_yticks([])

            ax.set_title(str(unit), fontsize=8)

        # -----------------------------
        # Remove empty subplots
        # -----------------------------
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle(stage, fontsize=16)

    plt.show()

def max_firing(tuning_curves, stage):
    
        max_firing = tuning_curves.max(dim= "head_direction")
        max_firing = pd.DataFrame({
        "stage": stage,
        "unit": max_firing.unit.values,
        "max_firing_rate": max_firing.values})
        return max_firing 

    
def tuning_width_fwhm(tuning_curves, stage):
    widths = {}

    angles = tuning_curves.coords["head_direction"].values
    dtheta = np.mean(np.diff(angles))  # angular bin size (radians)

    for unit in tuning_curves.coords["unit"].values:
        curve = tuning_curves.sel(unit=unit).values
        peak = curve.max()
        threshold = peak / 2

        # Boolean mask of bins above half max
        above = curve >= threshold

        # Handle circular wrap by doubling array
        above_double = np.concatenate([above, above])

        # Find longest contiguous stretch above threshold
        max_len = 0
        curr_len = 0
        for val in above_double:
            if val:
                curr_len += 1
                max_len = max(max_len, curr_len)
            else:
                curr_len = 0

        # Convert bins → radians
        width_rad = max_len * dtheta
        widths[unit] = width_rad
        
    df = pd.Series(widths, name="tuning_width_rad")
    df = df.reset_index()
    df.columns = ["unit", "tuning_width"]
    df["stage"] = stage
        
    return df
    
def avg_firing(NWB, stage):

    spikes = NWB["units"]
    epochs = NWB["epochs"]
    angle = NWB["ry"]

    # combine sleep epochs
    all_sleep = epochs[epochs.tags == "sleep1"].union(
                epochs[epochs.tags == "sleep2"])

    tuning_curves = nap.compute_tuning_curves(
        data=spikes,
        features=angle,
        bins=61,
        epochs=epochs[epochs.tags == "exploration"],
        feature_names=["head_direction"]
    )

    MI = nap.compute_mutual_information(tuning_curves)

    #restrict to HD cells
    HD_cells = MI.loc[MI["bits/spike"] > 0.4].index


    sleep_duration = all_sleep.tot_length("s")

    rates = []
    units = []

    for unit in HD_cells:

        spikes_unit = spikes[unit].restrict(all_sleep)

        n_spikes = len(spikes_unit)

        rate = n_spikes / sleep_duration

        units.append(unit)
        rates.append(rate)
        
    #data frame has two unit values, first column is HD cells # for that curve, second is the unit number from that recording
    df = pd.DataFrame({
        "unit": units,
        "avg_firing_rate": rates,
    })

    df["stage"] = stage

    return df

def continuous_stability(NWB, stage, plot_comparison=False):
    epochs = NWB["epochs"]
    spikes = NWB["units"]
    angle = NWB["ry"]

    # Restrict to exploration
    exploration = epochs[epochs.tags == "exploration"]

    # Compute full tuning curves
    tuning_full = nap.compute_tuning_curves(
        data=spikes,
        features=angle,
        bins=61,
        epochs=exploration,
        feature_names=["head_direction"]
    )

    # Identify HD cells
    MI = nap.compute_mutual_information(tuning_full)
    HD_cells = MI.loc[MI["bits/spike"] > 0.4].index

    # Split exploration in half
    t_start = exploration.start.min()
    t_end = exploration.end.max()
    t_mid = t_start + (t_end - t_start) / 2

    exploration_1 = nap.IntervalSet(start=t_start, end=t_mid)
    exploration_2 = nap.IntervalSet(start=t_mid, end=t_end)

    # Compute tuning curves for each half
    tuning_1 = nap.compute_tuning_curves(
        data=spikes, features=angle, bins=61, epochs=exploration_1, feature_names=["head_direction"]
    ).sel(unit=HD_cells)

    tuning_2 = nap.compute_tuning_curves(
        data=spikes, features=angle, bins=61, epochs=exploration_2, feature_names=["head_direction"]
    ).sel(unit=HD_cells)

    angles = tuning_1.coords["head_direction"].values
    corrs = {}

    # Compute Pearson correlation safely
    for unit in HD_cells:
        curve1 = tuning_1.sel(unit=unit).values
        curve2 = tuning_2.sel(unit=unit).values

        # Smooth curves
        curve1 = scipy.ndimage.gaussian_filter1d(curve1, sigma=2, mode='wrap')
        curve2 = scipy.ndimage.gaussian_filter1d(curve2, sigma=2, mode='wrap')

        # Mask NaNs/Infs
        mask = np.isfinite(curve1) & np.isfinite(curve2)
        if mask.sum() < 2:  # not enough valid points
            r = np.nan
        else:
            r, _ = scipy.stats.pearsonr(curve1[mask], curve2[mask])

        corrs[unit] = r

    # Build dataframe
    data = pd.Series(corrs, name="tuning_corr").reset_index()
    data.columns = ["unit", "tuning_corr"]
    data["stage"] = stage
    
    # Fisher r-to-z transform
    data["z"] = np.arctanh(data["tuning_corr"])

    # Optional plotting
    if plot_comparison:
        n_units = len(HD_cells)
        n_cols = 4
        n_rows = int(np.ceil(n_units / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        axes = axes.flatten()
        fig.subplots_adjust(wspace=0.4, hspace=0.6)

        for i, unit in enumerate(HD_cells):
            ax = axes[i]
            
            curve1 = scipy.ndimage.gaussian_filter1d(tuning_1.sel(unit=unit).values, sigma=2, mode='wrap')
            curve2 = scipy.ndimage.gaussian_filter1d(tuning_2.sel(unit=unit).values, sigma=2, mode='wrap')

            # Plot 2D curves
            ax.plot(angles, curve1, color="#56B5DA", linewidth=3, label='First Half')
            ax.plot(angles, curve2, color="#093B58", linewidth=3, label='Second Half')

            ax.set_title(f'Neuron {unit}', fontsize=8)
            ax.set_xlim(0, 2*np.pi)
            ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
            ax.set_xticklabels(["0", "90", "180", "270", "360"])
            ax.set_yticks([])  

            if i == 0:
                ax.legend(loc='upper right', fontsize=8)

        # Remove empty subplots
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle(stage, fontsize=16)
        plt.show()

    return data

def interleaved_stability(NWB, stage, plot_stability=False):
    epochs = NWB["epochs"]
    spikes = NWB["units"]
    angle = NWB["ry"]

    # Restrict to exploration
    exploration = epochs[epochs.tags == "exploration"]

    # Detect HD cells
    tuning_full = nap.compute_tuning_curves(
        data=spikes, features=angle, bins=61, epochs=exploration, feature_names=["head_direction"]
    )
    MI = nap.compute_mutual_information(tuning_full)
    HD_cells = MI.loc[MI["bits/spike"] > 0.4].index

    # Split exploration into 1-second bins
    bin_size = 1.0
    all_bins = []
    for start, end in zip(exploration.start, exploration.end):
        edges = np.arange(start, end, bin_size)
        for s in edges:
            e = min(s + bin_size, end)
            all_bins.append((s, e))

    all_bins = nap.IntervalSet(start=[b[0] for b in all_bins], end=[b[1] for b in all_bins])

    # Odd/even bins
    odd_bins = all_bins[::2]
    even_bins = all_bins[1::2]

    # Compute tuning curves
    tuning_odd = nap.compute_tuning_curves(
        data=spikes, features=angle, bins=61, epochs=odd_bins, feature_names=["head_direction"]
    ).sel(unit=HD_cells)

    tuning_even = nap.compute_tuning_curves(
        data=spikes, features=angle, bins=61, epochs=even_bins, feature_names=["head_direction"]
    ).sel(unit=HD_cells)

    angles = tuning_odd.coords["head_direction"].values
    corrs = {}

    for unit in HD_cells:
        curve_odd = tuning_odd.sel(unit=unit).values
        curve_even = tuning_even.sel(unit=unit).values

        # Smooth curves
        curve_odd = scipy.ndimage.gaussian_filter1d(curve_odd, sigma=2, mode='wrap')
        curve_even = scipy.ndimage.gaussian_filter1d(curve_even, sigma=2, mode='wrap')

        # Mask NaNs/Infs
        mask = np.isfinite(curve_odd) & np.isfinite(curve_even)
        if mask.sum() < 2:
            r = np.nan
        else:
            r, _ = scipy.stats.pearsonr(curve_odd[mask], curve_even[mask])

        corrs[unit] = r

    # Build dataframe
    data = pd.Series(corrs, name="tuning_corr").reset_index()
    data.columns = ["unit", "tuning_corr"]
    data["stage"] = stage
    
    # Fisher r-to-z transform
    data["z"] = np.arctanh(data["tuning_corr"])
    
    if plot_stability:
        n_units = len(HD_cells)
        n_cols = 4
        n_rows = int(np.ceil(n_units / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        axes = axes.flatten()
        fig.subplots_adjust(wspace=0.4, hspace=0.6)

        for i, unit in enumerate(HD_cells):
            ax = axes[i]
            
            curve1 = scipy.ndimage.gaussian_filter1d(tuning_even.sel(unit=unit).values, sigma=2, mode='wrap')
            curve2 = scipy.ndimage.gaussian_filter1d(tuning_odd.sel(unit=unit).values, sigma=2, mode='wrap')

            # Plot 2D curves
            ax.plot(angles, curve1, color="#56B5DA", linewidth=3, label='Odd Bins')
            ax.plot(angles, curve2, color="#093B58", linewidth=3, label='Even Bins')

            ax.set_title(f'Neuron {unit}', fontsize=8)
            ax.set_xlim(0, 2*np.pi)
            ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
            ax.set_xticklabels(["0", "90", "180", "270", "360"])
            ax.set_yticks([])  

            if i == 0:
                ax.legend(loc='upper right', fontsize=8)

        # Remove empty subplots
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle(stage, fontsize=16)
        plt.show()

    return data

def violin_plot(
    data_frame,
    y,
    ylabel=None,
    title=None,
    ref_line=None,
    run_stats=True, 
    get_desc_stats=True
):


    order = ["Proestrus", "Estrus", "Metestrus/Diestrus"]
    data_frame = data_frame.dropna(subset=[y])

    fig, ax = plt.subplots(figsize=(7,5))

    # -----------------------------
    # Violin plot
    # -----------------------------
    sns.violinplot(
        data=data_frame,
        x="stage",
        y=y,
        order=order,
        inner=None,
        cut=4,
        linewidth=1.2,
        width=0.8,
        palette=(["#CC79A7", "#54BBAB", "#56B4E9"]),
        ax=ax
    )

    sns.swarmplot(
        data=data_frame,
        x="stage",
        y=y,
        order=order,
        color="black",
        size=3.5,
        alpha=0.85,
        zorder=3,
        ax=ax
    )

    if ref_line is not None:
        ax.axhline(ref_line, color="black", linestyle="--", linewidth=1)

    ax.set_ylabel(ylabel if ylabel else y, fontsize=16, fontweight="bold")
    ax.set_xlabel("Estrous Cycle Stage", fontsize=16, fontweight="bold")
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")

    if get_desc_stats:
        # -----------------------------
        # Descriptive statistics
        # -----------------------------
        desc_stats = []

        for stage in order:
            values = data_frame.loc[data_frame.stage == stage, y]

            desc_stats.append({
                "Stage": stage,
                "N": len(values),
                "Mean": np.mean(values),
                "Median": np.median(values),
                "Std": np.std(values, ddof=1) if len(values) > 1 else np.nan,
                "SEM": stats.sem(values) if len(values) > 1 else np.nan,
                "Min": np.min(values) if len(values) > 0 else np.nan,
                "Max": np.max(values) if len(values) > 0 else np.nan
            })

        desc_df = pd.DataFrame(desc_stats)

        print("\nDescriptive Statistics:")
        print(desc_df.round(4))
        
    if run_stats: 
    # -----------------------------
    # Normality testing
    # -----------------------------
        normal = True
        groups_data = []

        for stage in order:
            values = data_frame.loc[data_frame.stage == stage, y]
            groups_data.append(values)

            if len(values) >= 3:
                _, p = stats.shapiro(values)
                if p < 0.05:
                    normal = False
            else:
                normal = False

        # -----------------------------
        # Store results
        # -----------------------------
        results = []

        # -----------------------------
        # Global test
        # -----------------------------
        if normal:
            model = ols(f'{y} ~ C(stage)', data=data_frame).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            global_p = anova_table["PR(>F)"][0]
            test_name = "ANOVA"

            results.append({
                "Test": "ANOVA",
                "Group 1": "All",
                "Group 2": "All",
                "Raw p-value": global_p,
                "Corrected p-value": np.nan
            })

        else:
            stat, global_p = stats.kruskal(*groups_data)
            test_name = "Kruskal-Wallis"

            results.append({
                "Test": "Kruskal-Wallis",
                "Group 1": "All",
                "Group 2": "All",
                "Raw p-value": global_p,
                "Corrected p-value": np.nan
            })

        # -----------------------------
        # Pairwise comparisons
        # -----------------------------
        pairs = list(itertools.combinations(order, 2))
        ymax = data_frame[y].max() 
        ymin = data_frame[y].min() 
        yrange = max(ymax - ymin, 0.05)

        if global_p >= 0.05:
            bar_height = ymax + 0.12 * yrange
            text_height = ymax + 0.15 * yrange
            ax.plot([0,2], [bar_height, bar_height], color="black", linewidth=1)
            ax.text(1, text_height, "n.s.", ha="center", fontsize=11)
            ax.set_ylim(ymin, ymax + 0.25*yrange)

        else:
            sig_pairs = []
            pvals_corrected = []

            if normal:
                # -----------------------------
                # Parametric: t-tests + Bonferroni
                # -----------------------------
                pvals = []

                for g1, g2 in pairs:
                    vals1 = data_frame.loc[data_frame.stage == g1, y]
                    vals2 = data_frame.loc[data_frame.stage == g2, y]

                    _, p = stats.ttest_ind(vals1, vals2)
                    pvals.append(p)

                pvals_corrected = np.minimum(np.array(pvals) * len(pvals), 1.0)

                for (g1, g2), raw_p, corr_p in zip(pairs, pvals, pvals_corrected):
                    results.append({
                        "Test": "t-test",
                        "Group 1": g1,
                        "Group 2": g2,
                        "Raw p-value": raw_p,
                        "Corrected p-value": corr_p
                    })

                    if corr_p < 0.05:
                        sig_pairs.append((g1, g2))

            else:
                # -----------------------------
                # Non-parametric: Dunn test with built in bonferroni
                # -----------------------------
                dunn = sp.posthoc_dunn(
                    data_frame,
                    val_col=y,
                    group_col="stage",
                    p_adjust="bonferroni"
                )

                for g1, g2 in pairs:
                    p_corr = dunn.loc[g1, g2]
                    pvals_corrected.append(p_corr)

                    results.append({
                        "Test": "Dunn",
                        "Group 1": g1,
                        "Group 2": g2,
                        "Raw p-value": np.nan,
                        "Corrected p-value": p_corr
                    })

                    if p_corr < 0.05:
                        sig_pairs.append((g1, g2))

            # -----------------------------
            # Annotation
            # -----------------------------
            if len(sig_pairs) > 0:
                annotator = Annotator(
                    ax,
                    sig_pairs,
                    data=data_frame,
                    x="stage",
                    y=y,
                    order=order
                )
                annotator.configure(test=None, text_format="star", loc="outside")

                annotator.set_pvalues([
                    pvals_corrected[i]
                    for i, pair in enumerate(pairs)
                    if pair in sig_pairs
                ])

                annotator.annotate()

            else:
                bar_height = ymax + 0.12 * yrange
                text_height = ymax + 0.15 * yrange
                ax.plot([0,2], [bar_height, bar_height], color="black", linewidth=1)
                ax.text(1, text_height, "n.s.", ha="center", fontsize=11)

            ax.set_ylim(ymin, ymax + 0.35*yrange)

        # -----------------------------
        # Print results table
        # -----------------------------
        results_df = pd.DataFrame(results)
        print("\nStatistical Results:")
        print(results_df)

    sns.despine()
    plt.tight_layout()
    plt.show()

'''MAIN BODY CODE'''

#plot tuning curves

md_tuning = [tuning_curves(nwb) for nwb in md_nwb]
md_tuning = [tc for tc in md_tuning if tc is not None]
#plot_curves(md_tuning, "Metestrus/Diestrus", "#56B4E9", polar = True)

e_tuning = [tuning_curves(nwb) for nwb in e_nwb]
e_tuning = [tc for tc in e_tuning if tc is not None]
#plot_curves(e_tuning, "Estrus", "#54BBAB", polar = True)

p_tuning = [tuning_curves(nwb) for nwb in p_nwb]
p_tuning = [tc for tc in p_tuning if tc is not None]
#plot_curves(p_tuning, "Proestrus", "#CC79A7", polar = True)

#Plot max firing 
md_max_firing = [max_firing(curve, "Metestrus/Diestrus") for curve in md_tuning]
e_max_firing = [max_firing(curve, "Estrus") for curve in e_tuning]
p_max_firing = [max_firing(curve, "Proestrus") for curve in p_tuning]


max_firing_df = pd.concat(e_max_firing + md_max_firing + p_max_firing, ignore_index = True)
print(max_firing_df)

violin_plot(
    data_frame = max_firing_df, 
    y = "max_firing_rate",
    ylabel = "Max Firing Rate (Hz)", 
    title = "ADN HD Cell Coding Property: Maximum Firing Rate During Exploration", 
    ref_line = None)

#Plot tuning width 
md_tuning_width = pd.concat([tuning_width_fwhm(curve, "Metestrus/Diestrus") for curve in md_tuning])
e_tuning_width = pd.concat([tuning_width_fwhm(curve, "Estrus") for curve in e_tuning])
p_tuning_width = pd.concat([tuning_width_fwhm(curve, "Proestrus") for curve in p_tuning])

tuning_widths_df = pd.concat([md_tuning_width, e_tuning_width, p_tuning_width], ignore_index=True)

violin_plot(
    data_frame = tuning_widths_df, 
    y = "tuning_width",
    ylabel = "Tuning Width (Radians)", 
    title = "ADN HD Cell Coding Property: Tuning Width (Full Width at Half Maximum)", 
    ref_line = None)

#plot avg firing
md_avg_firing = pd.concat([avg_firing(nwb, "Metestrus/Diestrus") for nwb in md_nwb])
e_avg_firing = pd.concat([avg_firing(nwb, "Estrus") for nwb in e_nwb])
p_avg_firing = pd.concat([avg_firing(nwb, "Proestrus") for nwb in p_nwb])

avg_firing_df = pd.concat(
    [md_avg_firing, e_avg_firing, p_avg_firing],
    ignore_index=True
)


violin_plot(
    data_frame = avg_firing_df, 
    y = "avg_firing_rate",
    ylabel = "Average Firing (Hz)", 
    title = "ADN HD Cell Coding Property: Average Firing During Sleep", 
    ref_line = None)

#plot continuous stability (can plot comparision plots if add True)

md_cont_stab = pd.concat(continuous_stability(nwb, "Metestrus/Diestrus") for nwb in md_nwb)
e_cont_stab = pd.concat(continuous_stability(nwb, "Estrus") for nwb in e_nwb)
p_cont_stab = pd.concat(continuous_stability(nwb, "Proestrus") for nwb in p_nwb)

cont_stab_df = pd.concat([md_cont_stab, e_cont_stab, p_cont_stab], ignore_index = True)

#Raw r values
violin_plot(
    data_frame = cont_stab_df, 
    y = "tuning_corr",
    ylabel = "Pearson Correlation (r)", 
    title = "ADN HD Cell Coding Property: Continuous Stability", 
    ref_line = 0.8,
    run_stats = False)

#Fisher r-to-z transformed 
violin_plot(
    data_frame = cont_stab_df, 
    y = "z",
    ylabel = "Fisher z-transformed Pearson correlation", 
    title = "ADN HD Cell Coding Property: Continuous Stability", 
    ref_line = 1, 
    get_desc_stats = False)
#plot interlevered stability 

md_inter_stab = pd.concat(interleaved_stability(nwb, "Metestrus/Diestrus") for nwb in md_nwb)
e_inter_stab = pd.concat(interleaved_stability(nwb, "Estrus") for nwb in e_nwb)
p_inter_stab = pd.concat(interleaved_stability(nwb, "Proestrus") for nwb in p_nwb)

inter_stab_df = pd.concat ([md_inter_stab,e_inter_stab, p_inter_stab ], ignore_index = True)

#Raw r values
violin_plot(
    data_frame = inter_stab_df, 
    y = "tuning_corr",
    ylabel = "Pearson Correlation (r)", 
    title = "ADN HD Cell Coding Property: Interlevered Stability", 
    ref_line = 0.8,
    run_stats = False)
    
#Fisher r-to-z transformed     
violin_plot(
    data_frame = inter_stab_df, 
    y = "z",
    ylabel = "Fisher z-transformed Pearson correlation", 
    title = "ADN HD Cell Coding Property: Interlevered Stability", 
    ref_line = 1, 
    get_desc_stats=False)