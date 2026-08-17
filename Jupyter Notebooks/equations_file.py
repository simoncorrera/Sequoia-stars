import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm, kstest
from hyppo.ksample import Energy
from itertools import combinations
import re
# File-path and dataset variables
galah_raw_allstar = '/Users/simoncorrera/Desktop/MQ NGC3201 project copy/fits and csv files/galah_dr4_allstar_240705.fits'
galah_raw_dynamics = '/Users/simoncorrera/Desktop/MQ NGC3201 project copy/fits and csv files/galah_dr4_vac_dynamics_240705.fits'

# galah_Gaia_csv = '/Users/simoncorrera/Desktop/MQ NGC3201 project copy/Jupyter Notebooks/Gahla_all_GAIA_kinematics.csv'

galah_Gaia_fits_original = '/Users/simoncorrera/Desktop/MQ NGC3201 project copy/fits and csv files/Galah_all_GAIA_kinematics' # THis is the same as the one directly below but this one does not include gaia ra, dec, pmra, pmdec, parallax, and radial velocity.
galah_Gaia_fits = '/Users/simoncorrera/Desktop/MQ NGC3201 project copy/fits and csv files/Gahla_all_GAIA_kinematics_Aug_2_using_previous_data.fits'

# Pradosh_csv = '/Users/simoncorrera/Desktop/MQ NGC3201 project copy/Jupyter Notebooks/Pradosh_all_GAIA_kinematics.csv'
Pradosh_fits = '/Users/simoncorrera/Desktop/MQ NGC3201 project copy/fits and csv files/Pradosh_all_GAIA_kinematics'
Pradosh_Revised_fits = '/Users/simoncorrera/Desktop/MQ NGC3201 project copy/fits and csv files/Pradosh_revision_kinematics.fits'

# ED_2_stream_csv = '/Users/simoncorrera/Desktop/MQ NGC3201 project copy/Jupyter Notebooks/ED_2__gaia_kinematics.csv'
ED_2_stream_fits = '/Users/simoncorrera/Desktop/MQ NGC3201 project copy/fits and csv files/ED_2__gaia_kinematics'

Vmans_NGC3201_txt = '/Users/simoncorrera/Desktop/MQ NGC3201 project copy/fits and csv files/clusters/catalogues/NGC_3201.txt'
Vmans_NGC5139_txt = '/Users/simoncorrera/Desktop/MQ NGC3201 project copy/fits and csv files/clusters/catalogues/NGC_5139_oCen.txt'
Vmans_NGC1851_txt = '/Users/simoncorrera/Desktop/MQ NGC3201 project copy/fits and csv files/clusters/catalogues/NGC_1851.txt'
Vmans_NGC0104_txt = '/Users/simoncorrera/Desktop/MQ NGC3201 project copy/fits and csv files/clusters/catalogues/NGC_104_47Tuc.txt'



def ensure_native_endian(df):
    """Convert all numeric NumPy columns in a DataFrame to native byte order."""
    for col in df.columns:
        dtype = df[col].dtype

        # Only handle real NumPy dtypes (skip pandas extension dtypes like StringDtype)
        if not isinstance(dtype, np.dtype):
            continue

        # Skip already-native or non-byte-order dtypes
        if dtype.byteorder in ('=', '|'):
            continue

        if np.issubdtype(dtype, np.number):
            df[col] = df[col].astype(dtype.newbyteorder('='))
    
    return df

"""----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"""

Vmans_NGC3201 = pd.read_csv(
    Vmans_NGC3201_txt,
    sep='\t',
    header=0
)
Vmans_NGC3201.columns = Vmans_NGC3201.columns.str.replace('#', '').str.strip()
Vmans_NGC3201_high_prob = Vmans_NGC3201[(Vmans_NGC3201['memberprob'] > 0.90)]# & (Vmans_NGC3201['qflag'] == 0)]



Vmans_NGC5139 = pd.read_csv(
    Vmans_NGC5139_txt,
    sep='\t',
    header=0
)
Vmans_NGC5139.columns = Vmans_NGC5139.columns.str.replace('#', '').str.strip()
Vmans_NGC5139_high_prob = Vmans_NGC5139[(Vmans_NGC5139['memberprob'] > 0.90)]# & (Vmans_NGC5139['qflag'] == 0)]


Vmans_NGC1851 = pd.read_csv(
    Vmans_NGC1851_txt,
    sep='\t',
    header=0
)
Vmans_NGC1851.columns = Vmans_NGC1851.columns.str.replace('#', '').str.strip()
Vmans_NGC1851_high_prob = Vmans_NGC1851[(Vmans_NGC1851['memberprob'] > 0.90)]# & (Vmans_NGC1851['qflag'] == 0)]


Vmans_NGC0104 = pd.read_csv(
    Vmans_NGC0104_txt,
    sep='\t',
    header=0
)
Vmans_NGC0104.columns = Vmans_NGC0104.columns.str.replace('#', '').str.strip()
Vmans_NGC0104_high_prob = Vmans_NGC0104[(Vmans_NGC0104['memberprob'] > 0.90)]# & (Vmans_NGC0104['qflag'] == 0)]

"""----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"""

def NGC3201_cuts_pradosh(df):  # I would need to make some small changes for this to work for galah. really Ijust need to change the RA DEC and Energy to the correct column name
      return((152 < df['RA']) & (156 > df['RA']) & (-48 < df['DEC']) & (-45 > df['DEC']) & (2000 > df['Energy']) & (-2500 > df['jphi']))

# Feuillet et al. 2021 https://doi.org/10.1093/mnras/stab2614
def Sequoia_cuts_Diane_2021(df):  
    return (-1.0< df['jphi']/df['jtot']) & (df['jphi']/df['jtot']<-0.6) & (-1.0 < (df['jz'] - df['jr'])/df['jtot']) & (-1.0 < (df['jz'] - df['jr'])/df['jtot']) & (0.1 > (df['jz'] - df['jr'])/df['jtot'])

def GSE_cuts_Diane_2021(df):
        return (-500< df['jphi']) & (500 > df['jphi']) & (30 < np.sqrt(df['jr'])) & (55 > np.sqrt(df['jr']))

"""----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"""

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.


    They weights are in effect first normalized so that they
    sum to 1 (and so they must not all be 0).


    values, weights -- NumPy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))

"""----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"""

def min_corner_plot_lables_comparison(df, number, acc_l, acc_r = False):
    ratio_columns = []
    labels_list = []
    e_labels_list = []
    accuracy_1 = []
    accuracy_2 = []

    if acc_r == False:
        for col in df.columns:
            if 'ratio' in col:
                ratio_columns.append(col)
        for i in range(number):
            labels  = []
            e_labels = []
            for ratio in ratio_columns:
                labels.append(df[ratio].iloc[i])
                e_labels.append('e_' + df[ratio].iloc[i])
            labels_list.append(labels)
            e_labels_list.append(e_labels)
            accuracy_1.append(df[acc_l].iloc[i])

        return(labels_list, e_labels_list, accuracy_1)
    else:
        for col in df.columns:
            if 'ratio' in col:
                ratio_columns.append(col)
        for i in range(number):
            labels  = []
            e_labels = []
            for ratio in ratio_columns:
                labels.append(df[ratio].iloc[i])
                e_labels.append('e_' + df[ratio].iloc[i])
            labels_list.append(labels)
            e_labels_list.append(e_labels)
            accuracy_1.append(df[acc_r].iloc[i])
            accuracy_2.append(df[acc_l].iloc[i])

        return(labels_list, e_labels_list, accuracy_1, accuracy_2)

"""----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"""

def ratio_totals_table(df, element_list, dimension):

    ratio_totals = pd.DataFrame(columns=['ratio', 'total'])

    for i in element_list:
        
        total = 0

        for n in range(1, dimension + 1): 
            ratio_col = f'ratio {n}'
            mask = (df[ratio_col] == i)
            total += mask.sum()

        # print(f'{i} N: {total}')
        ratio_totals.loc[len(ratio_totals)] = [i, total]
        ratio_totals = ratio_totals.sort_values(
        by='total',
        ascending=False
        )

    return ratio_totals

"""----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"""

def element_frequency(df_col_1, df_col_2, element_list):
    elm_frequency_table = pd.DataFrame(columns = ['element', 'frequency'])
    for elm in element_list:
        num = 0
        for ratio1 in df_col_1:
            if elm in (ratio1):
                num += 1
                # print(f'{elm} is in {ratio1}')
        for ratio2 in df_col_2:
            if elm in (ratio2):
                num += 1
                # print(f'{elm} is in {ratio1}')
        elm_frequency_table.loc[len(elm_frequency_table)] = [elm, num]
        elm_frequency_table = elm_frequency_table.sort_values(by = 'frequency', ascending=False)
        # print(f'{elm} was found {num} times')
    display(elm_frequency_table)
    return(elm_frequency_table)

"""----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"""

def top_min_corner_plots(df, number):
    top = df.head(number)
    return top

"""----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"""

def one_d_ks_test(ratio_list, object_list):
    (sample1_name, (df1,)), (sample2_name, (df2,)) = object_list.items()

    ratio_ks_test = pd.DataFrame(columns=['ratio', 'p_value', 'statistic', 'num_samples_1', 'num_samples_2'])
    for ratio in ratio_list:
        # plt.figure(figsize=(10, 6))
        # all_values = []


        # for key, (df) in object_list.items():

            # example datasets
        sample1 = df1[ratio].dropna()
        
        sample2 = df2[ratio].dropna()

        # perform KS test
        # statistic, p_value = ks_2samp(sample1, sample2)
        statistic, p_value = kstest(sample1, sample2)

        ratio_ks_test.loc[len(ratio_ks_test)] = [ratio, p_value, abs(statistic), len(sample1), len(sample2)]
        ratio_ks_test = ratio_ks_test.sort_values(
            by='p_value',
            ascending=True
            )
    return ratio_ks_test

"""----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"""

def two_d_ks_test(ratio_list, object_list, FE_H=False):
    (sample1_name, (df1,)), (sample2_name, (df2,)) = object_list.items()

    ratio_ks_test = pd.DataFrame(columns=['ratio 1', 'ratio 2', 'p_value', 'statistic', 'num_samples_1', 'num_samples_2'])

    if FE_H == True:
        if 'fe_h' in ratio_list:
            print('fe_h will be included')
    elif FE_H == False:
        ratio_list = [ratio for ratio in ratio_list if ratio != 'fe_h']

    for i in range(len(ratio_list)):
        for j in range(i + 1, len(ratio_list)):
            ratio1 = ratio_list[i]
            ratio2 = ratio_list[j]

            print("")
            print(f'{ratio1} and {ratio2}')    
            
            ratio1_tokens = {
                token
                for token in re.split(r'[_/]+', str(ratio1).lower())
                if token and re.fullmatch(r'[a-z]+', token)
            }
            ratio2_tokens = {
                token
                for token in re.split(r'[_/]+', str(ratio2).lower())
                if token and re.fullmatch(r'[a-z]+', token)
            }
            
            print(f'ratio1_tokens: {ratio1_tokens}')
            print(f'ratio2_tokens: {ratio2_tokens}')

            shared_segments = sorted(ratio1_tokens & ratio2_tokens)
            if shared_segments:
                print(f"Skipping {ratio1} and {ratio2}: shared segments = {shared_segments}")
                continue

            sample1 = df1[[ratio1, ratio2]].dropna()
            sample2 = df2[[ratio1, ratio2]].dropna()
            if len(sample1) <= 3 or len(sample2) <= 3:
                continue

            num_samples_1 = len(sample1)
            num_samples_2 = len(sample2)

            X = sample1.to_numpy()
            Y = sample2.to_numpy()

            statistic, p_value = Energy().test(X, Y)

            ratio_ks_test.loc[len(ratio_ks_test)] = [
                ratio1,
                ratio2,
                p_value,
                abs(statistic),
                num_samples_1,
                num_samples_2
            ]

    ratio_ks_test = ratio_ks_test.sort_values(by='p_value', ascending=True)
    return ratio_ks_test

"""----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"""

def three_d_ks_test(thee_d_ratio_list, object_list, FE_H = False):
    (sample1_name, (df1,)), (sample2_name, (df2,)) = object_list.items()

    ratio_ks_test = pd.DataFrame(columns=['ratio 1', 'ratio 2', 'ratio 3', 'p_value', 'statistic', 'num_samples_1', 'num_samples_2'])

    # if FE_H == True:
    #     if 'fe_h' in ratio_list:
    #         print('fe_h will be included')
    # elif FE_H == False:
    #     ratio_list = [ratio for ratio in ratio_list if ratio != 'fe_h']
    for i in range(len(thee_d_ratio_list)):

        print(f'{thee_d_ratio_list[i]}')

        # for key, (df) in object_list.items():
        ratio1 = thee_d_ratio_list[i][0]
        ratio2 = thee_d_ratio_list[i][1]
        ratio3 = thee_d_ratio_list[i][2]
            # example datasets
        # keep paired finite values
        sample1 = df1[[ratio1, ratio2, ratio3]].dropna()
        sample2 = df2[[ratio1, ratio2, ratio3]].dropna()
        num_samples_1 = len(sample1)
        num_samples_2 = len(sample2)
        
        # convert to numpy arrays
        X = sample1.to_numpy()
        Y = sample2.to_numpy()

        # multivariate comparison
        statistic, p_value = Energy().test(X, Y)

        ratio_ks_test.loc[len(ratio_ks_test)] = [
            ratio1,
            ratio2,
            ratio3,
            p_value,
            abs(statistic),
            num_samples_1,
            num_samples_2
        ]
    ratio_ks_test = ratio_ks_test.sort_values(
        by='p_value',
        ascending=True
        )
    return ratio_ks_test

"""----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"""

def fit_gmm_and_get_pdf(data, n_components, n_points=1000):
    """Fit GMM and return x + PDF"""
    data = data.dropna().values.reshape(-1, 1)

    gmm = GaussianMixture(n_components=n_components, random_state=42, n_init=20)
    gmm.fit(data)

    x = np.linspace(data.min(), data.max(), n_points).reshape(-1, 1)
    pdf = np.exp(gmm.score_samples(x))

    # return x, pdf
    return x, pdf, gmm

"""----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"""

def plot_gmm(data, label, color, n_components, maximum_peak = False, plot_compents = False, fontsize = 10, shift_right = 0.01, shift_up = 0.99, linewidth_main = 2, linewidth_components = 1.5):
    # """Plot GMM curve"""
    # x, pdf = fit_gmm_and_get_pdf(data, n_components)
    # plt.plot(x, pdf, lw=2, color=color, label=f"{label}")

    """Plot GMM curve + components"""

    x, pdf, gmm = fit_gmm_and_get_pdf(data, n_components)

    # Total mixture
    plt.plot(x, pdf, lw=linewidth_main, color=color, label=label)

    if maximum_peak:

        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_.flatten())
        weights = gmm.weights_

        print("Means:", means)
        print("Standard deviations:", stds)
        print("Weights:", weights)

        max_index = np.argmax(pdf)
        print("Maximum PDF value:", pdf[max_index])
        max_x = np.round(x[max_index][0], 3)
        if n_components == 1:
            # plt.text(shift_right, shift_up, f'{max_x:.2f} \u00B1 {stds[0]:.2f}', transform=plt.gca().transAxes, fontsize = fontsize, color=color, ha='left', va='top')
            if color == "#16b823d8":
                plt.text(max_x + shift_right, pdf[max_index] + 0.1 + shift_up, f'{max_x:.2f} \u00B1 {stds[0]:.2f}', fontsize = fontsize, color='g', ha='left', va='top')
            else:
                plt.text(max_x + shift_right, pdf[max_index] + 0.1 + shift_up, f'{max_x:.2f} \u00B1 {stds[0]:.2f}', fontsize = fontsize, color=color, ha='left', va='top')

        else:
            plt.text(shift_right, shift_up, f' L:{max_x}', transform=plt.gca().transAxes, fontsize = fontsize, color=color, ha='left', va='top')
        return(max_x)
    # Individual components
    components = sorted(
    zip(gmm.weights_, gmm.means_, gmm.covariances_),
    key=lambda x: x[1][0]
    )
    for i, (weight, mean, cov) in enumerate(components):

        sigma = np.sqrt(cov[0][0])

        component_pdf = weight * norm.pdf(
            x.flatten(),
            mean[0],
            sigma
        )
        if plot_compents:
            plt.plot(
            x.flatten(),
            component_pdf,
            color=color,
            ls='--',
            lw=linewidth_components,
            alpha=0.7
            )

            if maximum_peak == False:
                print(mean)
                mean_text = np.round(mean[0], 3)
                plt.text(shift_right, 0.98 - 0.05*i, f'C: {mean_text}', transform=plt.gca().transAxes, color=color, ha='left', va='top')

"""----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"""


def plot_hist(data, label, color, alpha):
    """Plot histogram"""
    plt.hist(data, bins=30, density=True, alpha=alpha, color=color, label=label)

"""----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"""

def find_ratio_combination_to_try_for_2d_KS_test(ratio_list, df1, df2, element_list, percent_value = 0.3, keep_num_elm = 6, do_costom_elm_filter = False, costom_elm_filter = []):
    
    # this will pick which ratios to filter by
    ratio_1d_ks_test = pd.DataFrame(columns=['ratio', 'p_value', 'statistic'])
    for ratio in ratio_list:
        sample1 = df1[ratio].dropna()
        sample2 = df2[ratio].dropna()

        statistic, p_value = kstest(sample1, sample2)

        ratio_1d_ks_test.loc[len(ratio_1d_ks_test)] = [ratio, p_value, statistic]
        ratio_1d_ks_test = ratio_1d_ks_test.sort_values(
            by='p_value',
            ascending=True
            )
    n = len(ratio_1d_ks_test)
    top_percent = ratio_1d_ks_test[:int(percent_value * n)]
    print('percent_value = ', percent_value)
    print('len(ratio_1d_ks_test) ', len(ratio_1d_ks_test))
    print('len(top_percent) ', len(top_percent))

    # Pick the elements to filter by
    if do_costom_elm_filter == False:
        elm_frequency_table = pd.DataFrame(columns = ['element', 'frequency', 'score'])
        for elm in element_list:
            num = 0
            for elm_ratio in top_percent['ratio']:
                if elm in elm_ratio:
                    num += 1
            score = 0
            for _, ratio_row in ratio_1d_ks_test.iterrows():
                if elm in ratio_row['ratio']:
                    print(f"{elm} and {ratio_row['ratio']}")
                    score += np.log(ratio_row['p_value'])
            elm_frequency_table.loc[len(elm_frequency_table)] = [elm, num, score]
            # elm_frequency_table = elm_frequency_table.sort_values(by = 'frequency', ascending=False)
        elm_frequency_table = elm_frequency_table.sort_values(by = 'score', ascending=True)


        display(elm_frequency_table)
        elm_filter = list(elm_frequency_table['element'][0:keep_num_elm])
        print('auto_elm_filter = ', elm_filter)
        # return(elm_frequency_table)
    if do_costom_elm_filter == True:
        elm_filter = costom_elm_filter
        print('costom_elm_filter = ', elm_filter)
    
    regex = '|'.join(map(re.escape, elm_filter))
    filtered_elm_list = top_percent[top_percent['ratio'].astype(str).str.contains(regex, na=False)].copy()
    print(len(filtered_elm_list))
    display(filtered_elm_list)
    return list(filtered_elm_list['ratio'])    

"""----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"""

# def make_2_vs_2_combinations(ratio_list, df1, df2):
def make_2_vs_2_combinations(ratio_list, df1, df2, element_list):


    new_cols_1_df1 = {}
    new_cols_1_df2 = {}
    combined_ratios_list = []
    for i in range(len(ratio_list)):
        ratio1 = ratio_list[i]
        sign_ratio_1 = np.sign(np.nanmean(df1[ratio1]) - np.nanmean(df2[ratio1]))
        for j in range(i + 1, len(ratio_list)):
            ratio2 = ratio_list[j]
            sign_ratio_2 = np.sign(np.nanmean(df1[ratio2]) - np.nanmean(df2[ratio2]))

            if sign_ratio_1 == sign_ratio_2:
                combined_ratio = f'{ratio1}_pl_{ratio2}'
                new_cols_1_df1[combined_ratio] = df1[ratio1] + df1[ratio2]
                new_cols_1_df2[combined_ratio] = df2[ratio1] + df2[ratio2]
            else:
                combined_ratio = f'{ratio1}_min_{ratio2}'
                new_cols_1_df1[combined_ratio] = df1[ratio1] - df1[ratio2]
                new_cols_1_df2[combined_ratio] = df2[ratio1] - df2[ratio2]
            combined_ratios_list.append(combined_ratio)

    df1 = pd.concat([df1, pd.DataFrame(new_cols_1_df1, index=df1.index)], axis=1)
    df2 = pd.concat([df2, pd.DataFrame(new_cols_1_df2, index=df2.index)], axis=1)
 
    print(f'len of combined_ratios_list is {len(combined_ratios_list)}')
    print(combined_ratios_list)


    combined_ratio_pairs = []
    for i in range(len(combined_ratios_list)):
        com_ratio_1 = combined_ratios_list[i]
        elms_in_com_ratio_1 = []
        for j in range(i+1, len(combined_ratios_list)):
            com_ratio_2 = combined_ratios_list[j]
            elms_in_com_ratio_2 = []
            for indi_elm in element_list:
                if indi_elm in com_ratio_1:
                    elms_in_com_ratio_1.append(indi_elm)
                if indi_elm in com_ratio_2:
                    elms_in_com_ratio_2.append(indi_elm)

            # convert to sets and check that there is no intersection
            set1 = set(elms_in_com_ratio_1)
            set2 = set(elms_in_com_ratio_2)
            if set1.isdisjoint(set2):
                combined_ratio_pairs.append([com_ratio_1, com_ratio_2])






    # for i in range(len(combined_ratios_list)):
    #     com_ratio_1 = combined_ratios_list[i]
    #     for j in range(i+1, len(combined_ratios_list)):
    #         com_ratio_2 = combined_ratios_list[j]
    #         for k in range(len(ratio_list)):
    #             for l in range(k+1, len(ratio_list)):
    #                 if (ratio_list[k] in com_ratio_1) and (ratio_list[l] in com_ratio_1) and (ratio_list[k] not in com_ratio_2) and (ratio_list[l] not in com_ratio_2):
    #                     combined_ratio_pairs.append([com_ratio_1, com_ratio_2])
    return(combined_ratio_pairs, df1, df2, combined_ratios_list)

"""----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"""
 
def n_one_vs_one_n(N, ratio_list, element_list):
    if N < 2 or N > len(ratio_list):
        return []

    combined_ratio_groups = []
    for combo in combinations(ratio_list, N):
        element_sets = []
        for ratio in combo:
            element_sets.append({elm for elm in element_list if elm in ratio})

        if all(
            element_sets[i].isdisjoint(element_sets[j])
            for i in range(len(element_sets))
            for j in range(i + 1, len(element_sets))
        ):
            combined_ratio_groups.append(list(combo))

    return combined_ratio_groups

"""----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"""

def make__2_vs_2_combinations_for_further_comparison(ratio_list, df3, combined_ratio_pairs):
    
    new_cols_1_df3 = {}
    combined_ratios_list = []
    for i in range(len(ratio_list)):
        ratio1 = ratio_list[i]
        for j in range(i + 1, len(ratio_list)):
            ratio2 = ratio_list[j]
            for k in range(len(combined_ratio_pairs)):
                comb_ratio = combined_ratio_pairs[k]
                if (ratio1 in comb_ratio) and (ratio2 in comb_ratio):
                    if 'pl' in comb_ratio:
                        new_cols_1_df3[comb_ratio] = df3[ratio1] + df3[ratio2]
                    if 'min' in comb_ratio:
                        new_cols_1_df3[comb_ratio] = df3[ratio1] - df3[ratio2]
    df3 = pd.concat([df3, pd.DataFrame(new_cols_1_df3, index=df3.index)], axis=1)
    return df3

"""----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"""

def make_elm_ratio_combinations_pradosh( data_set, element_list):
    element_in_list = element_list
    new_element_ratios_list = []
    new_element_ratio_errors_list = []
    data_set_copy = data_set

    # for i in element_in_list:
    #     if i in ["Ce", 'Zr', 'Nd']:
    #         data_set_copy = data_set_copy[data_set_copy['flag_'+i+'Fe'] == 0]

    for i in range(len(element_in_list)):
        if element_in_list[i] in ['Fe']:
            new_element_ratios_list.append('FeH')
            new_element_ratio_errors_list.append('e_FeH')
        for j in range(i+1, len(element_in_list)):
            if element_in_list[j] in ["Fe", "H"]:
                print("################  FE should be the first in the list, or H should be removed from the list ################")
            elif element_in_list[j] not in ["Fe", "H"]:
                col_name_for_numerator = element_in_list[j] + "Fe" # finds the names of the columns that will be needed to to calculate the numerator in the new ratio. E.g. if I want Ca/Ti, this would get Ca/Fe for the numerator
                # print("col_name_for_numerator:", col_name_for_numerator)
                err_col_numerator = 'e_' + col_name_for_numerator # same as above but now adds the 'e_' to find the error column for the corresponding ratio column
                # print('err_col_numerator:', err_col_numerator)

            if element_in_list[i] in [ "H"]:
                print("You shoulnd not include h")
            elif element_in_list[i] in ['Fe']:
                print('Fe is included')
                old_ratio_label = element_in_list[j] + "Fe"
                new_element_ratios_list.append(old_ratio_label)
                old_err_col = 'e_' + old_ratio_label
                new_element_ratio_errors_list.append(old_err_col)

            elif element_in_list[i] not in ["Fe", "H"]:
                col_name_for_denominator = element_in_list[i] + "Fe" # finds the names of the columns that will be needed to to calculate the denominator in the new ratio. E.g. if I want Ca/Ti, this would get Ti/Fe for the denominator
                # print('col_name_for_denominator:', col_name_for_denominator)
                err_col_denominator = 'e_' + col_name_for_denominator # same as above but now adds the 'e_' to find the error column for the corresponding ratio column
                # print('err_col_denominator:', err_col_denominator)
           
                new_ratio_label = element_in_list[j] + element_in_list[i]
                # print('new_ratio_label:', new_ratio_label)
                new_element_ratios_list.append(new_ratio_label)
                # print("new_element_ratios_list", new_element_ratios_list)
                # print("")

                new_ratio_error_label = 'e_' + element_in_list[j] + element_in_list[i]
                # print(new_ratio_error_label)
                new_element_ratio_errors_list.append(new_ratio_error_label)
                # print("new_element_ratio_errors_list", new_element_ratio_errors_list)
                # print("")

                data_set_copy[new_ratio_label] = data_set_copy[col_name_for_numerator] - data_set_copy[col_name_for_denominator] # this adds a new column to pradosh's catalog. It give the column the name and finds the ratio values
                data_set_copy[new_ratio_error_label] = np.sqrt((data_set_copy[err_col_numerator])**2 + (data_set_copy[err_col_denominator])**2)

    return data_set_copy, new_element_ratios_list, new_element_ratio_errors_list,

"""----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"""

def make_elm_ratio_combinations_galah( data_set, element_list):
    element_in_list = element_list
    new_element_ratios_list = []
    new_element_ratio_errors_list = []


    data_set_flagged = data_set
    for i in element_in_list:
        if i != 'fe':
            X_flag = 'flag_' + f'{i}' +'_fe'
            data_set_flagged = data_set_flagged[data_set_flagged[X_flag] == 0]

    for i in range(len(element_in_list)):
        if element_in_list[i] in ['fe']:
            new_element_ratios_list.append('fe_h')
            new_element_ratio_errors_list.append('e_fe_h')
        for j in range(i+1, len(element_in_list)):
            if element_in_list[j] in ["fe", "h"]:
                print("################  FE should be the first in the list, or H should be removed from the list ################")
            elif element_in_list[j] not in ["fe", "h"]:
                col_name_for_numerator = element_in_list[j] + "_fe" # finds the names of the columns that will be needed to to calculate the numerator in the new ratio. E.g. if I want Ca/Ti, this would get Ca/Fe for the numerator
                # print("col_name_for_numerator:", col_name_for_numerator)
                err_col_numerator = 'e_' + col_name_for_numerator # same as above but now adds the 'e_' to find the error column for the corresponding ratio column
                # print('err_col_numerator:', err_col_numerator)


            if element_in_list[i] in [ "h"]:
                print("You shoulnd not include h")
            elif element_in_list[i] in ['fe']:
                old_ratio_label = element_in_list[j] + "_fe"
                new_element_ratios_list.append(old_ratio_label)
                old_err_col = 'e_' + old_ratio_label
                new_element_ratio_errors_list.append(old_err_col)

            elif element_in_list[i] not in ["fe", "h"]:
                # print(element_in_list[i])
                col_name_for_denominator = element_in_list[i] + "_fe" # finds the names of the columns that will be needed to to calculate the denominator in the new ratio. E.g. if I want Ca/Ti, this would get Ti/Fe for the denominator
                # print('col_name_for_denominator:', col_name_for_denominator)
                err_col_denominator = 'e_' + col_name_for_denominator # same as above but now adds the 'e_' to find the error column for the corresponding ratio column
                # print('err_col_denominator:', err_col_denominator)
           
                new_ratio_label = element_in_list[j] + "_" + element_in_list[i]
                # print('new_ratio_label:', new_ratio_label)
                new_element_ratios_list.append(new_ratio_label)
                # print("new_element_ratios_list", new_element_ratios_list)
                # print("")


                new_ratio_error_label = 'e_' + element_in_list[j] + "_"+ element_in_list[i]
                # print('new_error_label:', new_ratio_error_label)
                new_element_ratio_errors_list.append(new_ratio_error_label)
                # print("new_element_ratio_errors_list", new_element_ratio_errors_list)
                # print("")
                # print('------------------------------------------------------')
                # print("")


                data_set_flagged[new_ratio_label] = data_set_flagged[col_name_for_numerator] - data_set_flagged[col_name_for_denominator] # this adds a new column to galah's catalog. It give the column the name and finds the ratio values
                data_set_flagged[new_ratio_error_label] = np.sqrt((data_set_flagged[err_col_numerator])**2 + (data_set_flagged[err_col_denominator])**2)
                data_set_flagged = data_set_flagged.dropna(subset=[new_ratio_label])


    return data_set_flagged, new_element_ratios_list, new_element_ratio_errors_list,

"""----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"""

def make_elm_ratio_combinations_galah_only_flag_nessessary_columns( data_set, element_list):
    element_in_list = element_list
    new_element_ratios_list = []
    new_element_ratio_errors_list = []
    data_set_copy = data_set.copy()

    for i in range(len(element_in_list)):
        if element_in_list[i] in ['fe']:
            new_element_ratios_list.append('fe_h')
            new_element_ratio_errors_list.append('e_fe_h')


        for j in range(i+1, len(element_in_list)):
            if element_in_list[j] in ["fe", "h"]:
                print("################  FE should be the first in the list, or H should be removed from the list ################")
            elif element_in_list[j] not in ["fe", "h"]:
                col_name_for_numerator = element_in_list[j] + "_fe"
                err_col_numerator = 'e_' + col_name_for_numerator

            if element_in_list[i] in ["h"]:
                print("You shoulnd not include h")
            elif element_in_list[i] in ['fe']:
                old_ratio_label = element_in_list[j] + "_fe"
                new_element_ratios_list.append(old_ratio_label)
                old_err_col = 'e_' + old_ratio_label
                new_element_ratio_errors_list.append(old_err_col)

            elif element_in_list[i] not in ["fe", "h"]:
                col_name_for_denominator = element_in_list[i] + "_fe"
                err_col_denominator = 'e_' + col_name_for_denominator

                new_ratio_label = element_in_list[j] + "_" + element_in_list[i]
                new_element_ratios_list.append(new_ratio_label)

                new_ratio_error_label = 'e_' + element_in_list[j] + "_"+ element_in_list[i]
                new_element_ratio_errors_list.append(new_ratio_error_label)

                if new_ratio_label not in data_set_copy.columns:
                    data_set_copy[new_ratio_label] = np.nan
                    data_set_copy[new_ratio_error_label] = np.nan

                valid_mask = pd.Series(True, index=data_set_copy.index)
                if element_in_list[i] != 'fe':
                    x1_flag = 'flag_' + f'{element_in_list[i]}' + '_fe'
                    valid_mask &= (data_set_copy[x1_flag] == 0)
                if element_in_list[j] != 'fe':
                    x2_flag = 'flag_' + f'{element_in_list[j]}' + '_fe'
                    valid_mask &= (data_set_copy[x2_flag] == 0)

                data_set_copy.loc[:, new_ratio_label] = np.nan
                data_set_copy.loc[:, new_ratio_error_label] = np.nan
                data_set_copy.loc[valid_mask, new_ratio_label] = (
                    data_set_copy.loc[valid_mask, col_name_for_numerator] - data_set_copy.loc[valid_mask, col_name_for_denominator]
                )
                data_set_copy.loc[valid_mask, new_ratio_error_label] = np.sqrt(
                    (data_set_copy.loc[valid_mask, err_col_numerator])**2 + (data_set_copy.loc[valid_mask, err_col_denominator])**2
                )

    return data_set_copy, new_element_ratios_list, new_element_ratio_errors_list,

