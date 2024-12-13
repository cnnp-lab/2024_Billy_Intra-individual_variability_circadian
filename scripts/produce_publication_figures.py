import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os

STAT_ROUND = 3
ALPHA = 0.05

CONTROL_COLOR = "tab:blue"
PWE_COLOR = "tab:orange"

def results1(segments_df):

	# calculate intra-individual average (IIA) and variability (IIV) by averaging across all segments for each subject
	figure_df = segments_df.groupby("Participant")[
		["Period Mean", "Period Std", "Acrophase Mean", "Acrophase Std", "Amplitude Mean", "Amplitude Std"]].mean()
	figure_df = figure_df.rename(columns={"Period Mean": "Period IIA",
					  "Period Std": "Period IIV",
					  "Acrophase Mean": "Acrophase IIA",
					 "Acrophase Std": "Acrophase IIV",
					 "Amplitude Mean": "Amplitude IIA",
					 "Amplitude Std": "Amplitude IIV"})
	# acrophase is a circular variable, so requires circular mean calculation
	figure_df["Acrophase IIA"] = segments_df.groupby("Participant")["Acrophase Mean"].apply(stats.circmean, high=23.99, low=0.00)
	figure_df["Cohort"] = figure_df.index.map(lambda x: "PWE" if "PWE" in x else "Control")
	figure_df[["Total Duration (h)", "N Seizures Total"]] = segments_df.groupby("Participant")[["Duration (h)", "N Seizures"]].sum()
	figure_df["Total Duration (days)"] = figure_df["Total Duration (h)"] / 24

	def panel(ax, property):

		sns.swarmplot(data=figure_df, x="Cohort", y=property, hue="Cohort",
		 			  			  hue_order=["Control", "PWE"], zorder=0.1, size=3, ax=ax)

		sns.boxplot(data=figure_df, x="Cohort", y=property, hue="Cohort", hue_order=["Control", "PWE"],
					showfliers=False, width=0.6, linewidth=1,
					boxprops={"alpha":0.25},
					medianprops={"color": "black", "alpha":0.75},
					whiskerprops={"alpha":0.75}, capprops={"alpha":0}, ax=ax)


		# adjust y limits to zoom-in on main distribution of data
		Q1 = figure_df[property].quantile(0.25)
		Q3 = figure_df[property].quantile(0.75)
		IQR = Q3-Q1
		lower_threshold = Q1 - 1.5 * IQR
		upper_threshold = Q3 + 1.5 * IQR

		values_between_threshold =  figure_df[(figure_df[property] >= lower_threshold) & (figure_df[property] <= upper_threshold)][property].values

		lower_nearest_value = values_between_threshold[np.nanargmin(np.abs(values_between_threshold - lower_threshold))]
		upper_nearest_value = values_between_threshold[np.nanargmin(np.abs(values_between_threshold - upper_threshold))]

		buffer = (upper_nearest_value - lower_nearest_value)* 0.15

		ax.set_ylim(lower_nearest_value - buffer, upper_nearest_value + buffer)


		# perform wilcoxon rank-sum
		z_stat, p_value = stats.ranksums(figure_df[figure_df["Cohort"] == "PWE"][property],
							  figure_df[figure_df["Cohort"] == "Control"][property],
							  alternative="two-sided")
		test_label = f"z={np.round(z_stat, STAT_ROUND)}, p={np.round(p_value, STAT_ROUND)}"

		if "IIA" in property:
			ax.set_title("Intra-Individual Average")
			ax.set_xlabel(test_label, weight="bold" if p_value < ALPHA else "regular")
		elif "IIV" in property:
			ax.set_title("Intra-Individual Variability")
			ax.set_xlabel(test_label, weight="bold" if p_value < ALPHA else "regular")
			ax.yaxis.set_ticks_position("right")
			ax.yaxis.set_label_position("right")

		ax.set(xticklabels=[])
		ax.tick_params(bottom=False)

	# produce result figure
	fig, axs = plt.subplots(3, 2, figsize=(4, 7))

	panel(axs[0, 0], "Period IIA")
	panel(axs[1, 0], "Acrophase IIA")
	panel(axs[2, 0], "Amplitude IIA")

	panel(axs[0, 1], "Period IIV")
	panel(axs[1, 1], "Acrophase IIV")
	panel(axs[2, 1], "Amplitude IIV")

	axs[0, 0].set_ylabel("Hours")
	axs[0, 1].set_ylabel("Hours")

	axs[1, 0].set_ylabel("Time of Day [h]")
	axs[1, 0].get_yaxis().set_major_formatter(
		matplotlib.ticker.FuncFormatter(lambda x, p: f"{int(x)}"))
	axs[1, 1].set_ylabel("Hours")

	axs[2, 0].set_ylabel("Beats per minute")
	axs[2, 1].set_ylabel("Beats per minute")

	handles = [
		matplotlib.lines.Line2D([0], [0],
					 marker='o',
					 color=CONTROL_COLOR,
					 markerfacecolor=CONTROL_COLOR,
					 markersize=5,
					 linestyle='',
					 label=f'Controls (n={len(figure_df[figure_df["Cohort"] == "Control"])})'),
		matplotlib.lines.Line2D([0], [0],
					 marker='o',
					 color=PWE_COLOR,
					 markerfacecolor=PWE_COLOR,
					 markersize=5,
					 linestyle='',
					 label=f'PWE (n={len(figure_df[figure_df["Cohort"] == "PWE"])})'),

	]
	fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.99))

	fig.text(0.5, 0.895, "Period", ha="center", fontsize=14)
	fig.text(0.5, 0.590, "Acrophase", ha="center", fontsize=14)
	fig.text(0.5, 0.290, "Amplitude", ha="center", fontsize=14)

	plt.tight_layout(rect=[0, 0, 1, 0.90], h_pad=2.5)

	return fig, axs


def results2(segments_df):

	# filter to weeks occurring within the time period they were recording their seizures
	segments_df = segments_df[segments_df["In Seizure Diary"] == True]

	figure_df = segments_df.groupby("Participant")[
		["Period Mean", "Period Std", "Acrophase Mean", "Acrophase Std", "Amplitude Mean", "Amplitude Std"]].mean()
	figure_df = figure_df.rename(columns={"Period Mean": "Period IIA",
										  "Period Std": "Period IIV",
										  "Acrophase Mean": "Acrophase IIA",
										  "Acrophase Std": "Acrophase IIV",
										  "Amplitude Mean": "Amplitude IIA",
										  "Amplitude Std": "Amplitude IIV"})
	figure_df["Acrophase IIA"] = segments_df.groupby("Participant")["Acrophase Mean"].apply(stats.circmean, high=23.99,
																							low=0.00)
	figure_df["Cohort"] = figure_df.index.map(lambda x: "PWE" if "PWE" in x else "Control")
	figure_df[["Total Duration (h)", "N Seizures Total"]] = segments_df.groupby("Participant")[
		["Duration (h)", "N Seizures"]].sum()
	figure_df["Total Duration (days)"] = figure_df["Total Duration (h)"] / 24

	# calculate seizure frequency for each PWE
	figure_df["Seizure Frequency (seizures/week)"] = figure_df["N Seizures Total"] / (figure_df["Total Duration (days)"] / 7)

	# discard any subjects with no seizures occurring, as this messes with the correlation
	figure_df = figure_df[figure_df["N Seizures Total"] > 0]

	# outlier removal
	figure_df = figure_df[np.abs(stats.zscore(figure_df["Period IIV"])) < 3]
	figure_df = figure_df[np.abs(stats.zscore(figure_df["Acrophase IIV"])) < 3]
	figure_df = figure_df[np.abs(stats.zscore(figure_df["Amplitude IIV"])) < 3]

	Y = "Seizure Frequency (seizures/week)"

	def panel(ax, property):

		# calculate the regression on the log of seizure frequency
		slope, intercept, rvalue, pvalue, stderr = stats.linregress(figure_df[property], np.log10(figure_df[Y]))

		# plot in linear non-log scale
		ax.scatter(figure_df[property], figure_df[Y], color=PWE_COLOR, alpha=0.5)

		reg_line_x = figure_df.sort_values(by=property)[property].to_numpy()  # sort the x-axis values so line plots nicely
		# IDK why we have to raise the reg line to power 10?
		ax.plot(reg_line_x, 10**(slope * reg_line_x + intercept), color="black")

		# rather than convert sz/freq values to log, we set y axis to log scale, which formats ticks better
		ax.set_yscale("log", base=10)
		# also, we can convert the values to readable sz/week values (1 sz/week rather than 10^0)
		ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

		p_label = f"{np.round(pvalue, STAT_ROUND)}"
		p_weight = "bold" if pvalue < ALPHA else "normal"
		ax.set_title(
			f"$R^2={np.round(np.power(rvalue, 2), STAT_ROUND)}$, $r={np.round(rvalue, STAT_ROUND)}$,  $p$ = {p_label}", weight=p_weight)

		ax.set_xlabel(f"{property} [{'bpm' if property == 'Amplitude' else 'h'}]")


	fig, axs = plt.subplots(3, 1, figsize=(3.5, 7))

	panel(axs[0], "Period IIV")
	panel(axs[1], "Acrophase IIV")
	panel(axs[2], "Amplitude IIV")

	for ax in axs.flatten():
		ax.set_ylabel(Y)


	legend_elements = [matplotlib.patches.Patch(facecolor=PWE_COLOR, edgecolor="black",
							 label=f"PWE (n={len(figure_df)})")]
	axs[0].legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 1.4))

	plt.tight_layout()

	return fig, axs

def results3(segments_df):

	# for PWE, filter to weeks occuring within the time period they were recording their seizures. keep controls too.
	segments_df = segments_df[(segments_df["In Seizure Diary"] == True) | (segments_df["Cohort"] == "Control")]

	# create boolean for whether seizures occurred during that segment
	segments_df["Seizures"] = segments_df["N Seizures"] > 0

	# calculate intra-individual average (IIA) and variability (IIV) by averaging across all segments for each subject
	# however, do it separately for weeks where seizures did occur and weeks where seizures did not occur
	# (producing 2 rows per PWE)
	figure_df = segments_df.groupby(["Participant", "Seizures"])[
		["Period Mean", "Period Std", "Acrophase Mean", "Acrophase Std", "Amplitude Mean", "Amplitude Std"]].mean()
	figure_df = figure_df.rename(columns={"Period Mean": "Period IIA",
										  "Period Std": "Period IIV",
										  "Acrophase Mean": "Acrophase IIA",
										  "Acrophase Std": "Acrophase IIV",
										  "Amplitude Mean": "Amplitude IIA",
										  "Amplitude Std": "Amplitude IIV"})
	figure_df["Acrophase IIA"] = segments_df.groupby(["Participant", "Seizures"])["Acrophase Mean"].apply(stats.circmean, high=23.99,
																							low=0.00)
	figure_df = figure_df.reset_index()  # split subject/in_sz_diary multi-index back into columns

	figure_df["Cohort"] = figure_df["Participant"].map(lambda x: "PWE" if "PWE" in x else "Control")
	figure_df["N Segments"] = figure_df.apply(
		lambda x: segments_df.groupby(["Participant", "Seizures"]).size().loc[(x["Participant"], x["Seizures"])], axis=1)
	figure_df[["Total Duration (h)", "N Seizures Total"]] = segments_df.groupby(["Participant", "Seizures"])[
		["Duration (h)", "N Seizures"]].sum().reset_index()[["Duration (h)", "N Seizures"]]
	figure_df["Total Duration (days)"] = figure_df["Total Duration (h)"] / 24

	# drop PWE without at least 5 seizure-containing and 5 seizure-free weeks
	pwe_to_keep = []
	for subject in figure_df["Participant"].unique():
		subject_df = figure_df[figure_df["Participant"] == subject]

		if len(subject_df) == 2 and all(subject_df.apply(lambda x: x["N Segments"] >= 5, axis=1)):
			pwe_to_keep.append(subject)
	figure_df = figure_df[(figure_df["Cohort"] == "Control") | (figure_df["Participant"]).isin(pwe_to_keep)]

	figure_df["Cohort"] = figure_df.apply(
		lambda x: f"PWE\n{'(Seizure-containing)' if x['Seizures'] else '(Seizure-free)'}"
		if x["Cohort"] != "Control" else x["Cohort"], axis=1)

	hue_order = ["Control", "PWE\n(Seizure-free)", "PWE\n(Seizure-containing)"]
	palette = [sns.color_palette()[i] for i in [0, 1, 1]]

	szcont = figure_df[figure_df["Cohort"] == "PWE\n(Seizure-containing)"].set_index("Participant")
	szfree = figure_df[figure_df["Cohort"] == "PWE\n(Seizure-free)"].set_index("Participant")

	def panel(ax, property):
		sns.swarmplot(data=figure_df, x="Cohort", y=property, hue="Cohort", hue_order=hue_order,
					  palette=palette,
					  zorder=0.1, size=3, ax=ax)

		sns.boxplot(data=figure_df, x="Cohort", y=property, hue="Cohort", hue_order=hue_order,
					palette=palette,
					showfliers=False, width=0.6, linewidth=1,
					boxprops={"alpha": 0.25},
					medianprops={"color": "black", "alpha": 0.75},
					whiskerprops={"alpha": 0.75}, capprops={"alpha": 0}, ax=ax)



		# adjust y limits to zoom-in on main distribution of data
		Q1 = figure_df[property].quantile(0.25)
		Q3 = figure_df[property].quantile(0.75)
		IQR = Q3 - Q1
		lower_threshold = Q1 - 1.5 * IQR
		upper_threshold = Q3 + 1.5 * IQR

		values_between_threshold = figure_df[
			(figure_df[property] >= lower_threshold) & (figure_df[property] <= upper_threshold)
			][property].values

		lower_nearest_value = values_between_threshold[np.nanargmin(np.abs(values_between_threshold - lower_threshold))]
		upper_nearest_value = values_between_threshold[np.nanargmin(np.abs(values_between_threshold - upper_threshold))]

		buffer = (upper_nearest_value - lower_nearest_value) * 0.15

		ax.set_ylim(lower_nearest_value - buffer, upper_nearest_value + buffer)


		if not all(szcont.index == szfree.index):
			raise ValueError("Data is not paired! Wilcoxon signed-rank invalid ")

		# perform wilcoxon signed-rank
		test = stats.wilcoxon(szcont[property], szfree[property], alternative="two-sided", method="approx")
		z_stat = test.zstatistic
		p_value = test.pvalue
		test_label = f"z={np.round(z_stat, STAT_ROUND)}, p={np.round(p_value, STAT_ROUND)}"

		if "IIA" in property:
			ax.set_title("Intra-Individual Average")
			ax.set_xlabel(test_label,
						  weight="bold" if p_value < ALPHA else "regular")
		elif "IIV" in property:
			ax.set_title("Intra-Individual Variability")
			ax.set_xlabel(test_label,
						  weight="bold" if p_value < ALPHA else "regular")
			ax.yaxis.set_ticks_position("right")
			ax.yaxis.set_label_position("right")

		ax.set(xticklabels=[])
		ax.tick_params(bottom=False)

		# modify markers for PWE seizure-containing
		for artist, hue_level in zip(ax.collections, hue_order):
			if hue_level == "PWE\n(Seizure-free)":
				artist.set_facecolor('none')
				artist.set_linewidth(0.5)  # by default, linwidth=0, so won't appear
				artist.set_edgecolor(palette[1])



	fig, axs = plt.subplots(3, 2, figsize=(4, 7))

	panel(axs[0, 0], "Period IIA")
	panel(axs[1, 0], "Acrophase IIA")
	panel(axs[2, 0], "Amplitude IIA")

	panel(axs[0, 1], "Period IIV")
	panel(axs[1, 1], "Acrophase IIV")
	panel(axs[2, 1], "Amplitude IIV")

	axs[0, 0].set_ylabel("Hours")
	axs[0, 1].set_ylabel("Hours")

	axs[1, 0].set_ylabel("Time of Day [h]")
	axs[1, 0].get_yaxis().set_major_formatter(
		matplotlib.ticker.FuncFormatter(lambda x, p: f"{int(x)}"))
	axs[1, 1].set_ylabel("Hours")

	axs[2, 0].set_ylabel("Beats per minute")
	axs[2, 1].set_ylabel("Beats per minute")

	handles = [
		matplotlib.lines.Line2D([0], [0], marker='o',
					 color=palette[0],
					 markerfacecolor=palette[0],
					 markersize=5,
					 linestyle='',
					 label=f'Controls (n={len(figure_df[figure_df["Cohort"] == "Control"])})'),
		matplotlib.lines.Line2D([0], [0],
					 marker='o',
					 color=palette[1],
					 markerfacecolor='none',
					 markersize=5,
					 linestyle='',
					 label=f'PWE (Seizure-free) (n={len(szfree)})'),
		matplotlib.lines.Line2D([0], [0],
					 marker='o',
					 color=palette[2],
					 markerfacecolor=palette[2],
					 markersize=5,
					 linestyle='',
					 label=f'PWE (Seizure-containing) (n={len(szcont)})')
	]
	fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.99))

	fig.text(0.5, 0.865, "Period", ha="center", fontsize=14)
	fig.text(0.5, 0.575, "Acrophase", ha="center", fontsize=14)
	fig.text(0.5, 0.275, "Amplitude", ha="center", fontsize=14)

	plt.tight_layout(rect=[0, 0, 1, 0.87], h_pad=2.5)

	return fig, axs

if __name__ == "__main__":
	"""
	"""

	# initialise
	plt.rc("font", family="helvetica")
	plt.rcParams.update({'font.size': 8, })
	os.makedirs("figures", exist_ok=True)

	# load in dataframe of circadian property mean/std per segment across all participants
	segments_df = pd.read_csv("../data/segments_df.csv", index_col=0)

	# calculate intra-individual average and variabilities and produce result figures
	fig1, axs1 = results1(segments_df)
	fig2, axs2 = results2(segments_df)
	fig3, axs3 = results3(segments_df)

	fig1.show()
	fig2.show()
	fig3.show()

	fig1.savefig("../figures/results1.pdf", backend="cairo")
	fig2.savefig("../figures/results2.pdf", backend="cairo")
	fig3.savefig("../figures/results3.pdf", backend="cairo")

