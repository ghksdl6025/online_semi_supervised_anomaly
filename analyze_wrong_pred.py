#%%
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

#%%
df = pd.read_excel('./analyze_why_happend.xlsx',sheet_name='Original')
df_why = pd.read_excel('./analyze_why_happend.xlsx',sheet_name='Why')
prefix_13 = df[df['prefix']==13].reset_index(drop=True)
prefix_13_why = df_why[df_why['prefix']==13].reset_index(drop=True)
print(classification_report(prefix_13['true_noise'], prefix_13['pred_ad_anom']))
print(classification_report(prefix_13_why['true_noise'], prefix_13_why['pred_ad_anom']))


#%%
true_noise = list(prefix_13_why['true_noise'])
pred_anom = list(prefix_13_why['pred_ad_anom'])

comp_ad_result =dict()
comp_ad_result['both normal'] = []
comp_ad_result['normal anomal'] = []
comp_ad_result['both anomal'] = []
comp_ad_result['anomal normal'] = []

for pos, i in enumerate(true_noise):
    if i ==0 and pred_anom[pos] ==0:
        comp_ad_result['both normal'].append(pos)
    elif i==0 and pred_anom[pos] ==1:
        comp_ad_result['normal anomal'].append(pos)
    elif i==1 and pred_anom[pos] ==1:
        comp_ad_result['both anomal'].append(pos)
    elif i ==1 and pred_anom[pos] ==0:
        comp_ad_result['anomal normal'].append(pos)



#%%
print('Both normal:', len(comp_ad_result['both normal']))
print('Normal anomal:', len(comp_ad_result['normal anomal']))
print('Both anomal:', len(comp_ad_result['both anomal']))
print('Anormal nomal:',len(comp_ad_result['anomal normal']))

#%%
import ast
import matplotlib.pyplot as plt
normal_normal = prefix_13_why.iloc[comp_ad_result['both normal'],:]
normal_anomal = prefix_13_why.iloc[comp_ad_result['normal anomal'],:]
anomal_anomal = prefix_13_why.iloc[comp_ad_result['both anomal'],:]
anomal_normal = prefix_13_why.iloc[comp_ad_result['anomal normal'],:]
# for i in list(normal_anomal['nap_prob']):
#     i = ast.literal_eval(i)
#     print(type(i))
#     print(np.argmax(i))

# print(normal_anomal['prob_ad'].describe())
# print(normal_normal['prob_ad'].describe())
# print(anomal_normal['prob_ad'].describe())
# print(anomal_anomal['prob_ad'].describe())
print(type(normal_anomal['prob_ad']))
# plt.hist(normal_normal['prob_ad'], bins=25, label = 'Both normal')
plt.hist(normal_anomal['prob_ad'], bins=25, label = 'Normal Anomal')
plt.hist(anomal_anomal['prob_ad'], bins=25, label = 'Both Anomal')
# plt.hist(anomal_normal['prob_ad'], bins=25, label = 'Anomal normal')
plt.xlim(0.5, 0.8)
plt.legend()
plt.show()
# %%

noise_df = pd.read_csv('./data/0.049_noise.csv')
normal_normal = noise_df[noise_df['Case ID'].isin(list(normal_normal['case_id']))]

#%%
prefix13_noise = pd.read_csv('./prefix13_noise0.049.csv')
normal_normal.to_csv('./prefix13_noise0.049_nomal_nomal.csv',index=False)
groups = normal_normal.groupby('Case ID')
concating = []
for _, group in groups:
    group = group.reset_index(drop=True)
    group = group.iloc[:13,:]
    concating.append(group)

pd.concat(concating).to_csv('./prefix13_noise0.049_nomal_nomalv2.csv',index=False)

# %%
from sklearn.metrics import accuracy_score, hamming_loss, roc_auc_score, f1_score


prefix_nap_acc = dict()
anomal_f1 =[]
for prefix in range(2,14):

    result_49 = pd.read_csv('./result/0.049_noise.csv_classifier_xgb_50_random_sample.csv')
    # result_49 = pd.read_csv('./result/0.099_noise.csv_fixed_v2.csv')
    # result_49 = pd.read_csv('./result/0.049_noise.csv_classifier_xgb_50_random_sample_napv2.csv')


    prefix_13_49 = result_49[result_49['prefix']==prefix]

    y_true = list(prefix_13_49['actual_act'])
    y_pred = [ast.literal_eval(i)[0] for i in list(prefix_13_49['predict_act'])]
    total_acc = accuracy_score(y_true, y_pred)
    anomal_f1 = classification_report(prefix_13_49['true_noise'], prefix_13_49['pred_ad_anom'],output_dict=True)['1']['f1-score']

    prefix_13_49 = prefix_13_49[prefix_13_49['true_noise']==0]
    prefix_13_49 = prefix_13_49[prefix_13_49['pred_ad_anom']==0]
    y_true = list(prefix_13_49['actual_act'])
    y_pred = [ast.literal_eval(i)[0] for i in list(prefix_13_49['predict_act'])]
    normal_normal_acc = accuracy_score(y_true, y_pred)
    prefix_13_49 = result_49[result_49['prefix']==prefix]
    prefix_13_49 = prefix_13_49[prefix_13_49['true_noise']==0]
    prefix_13_49 = prefix_13_49[prefix_13_49['pred_ad_anom']==1]
    y_true = list(prefix_13_49['actual_act'])
    y_pred = [ast.literal_eval(i)[0] for i in list(prefix_13_49['predict_act'])]
    normal_anomal_acc = accuracy_score(y_true, y_pred)

    prefix_13_49 = result_49[result_49['prefix']==prefix]
    prefix_13_49 = prefix_13_49[prefix_13_49['true_noise']==1]
    prefix_13_49 = prefix_13_49[prefix_13_49['pred_ad_anom']==0]
    y_true = list(prefix_13_49['actual_act'])
    y_pred = [ast.literal_eval(i)[0] for i in list(prefix_13_49['predict_act'])]

    anomal_normal_acc = accuracy_score(y_true, y_pred)

    prefix_13_49 = result_49[result_49['prefix']==prefix]
    prefix_13_49 = prefix_13_49[prefix_13_49['true_noise']==1]
    prefix_13_49 = prefix_13_49[prefix_13_49['pred_ad_anom']==1]
    y_true = list(prefix_13_49['actual_act'])
    y_pred = [ast.literal_eval(i)[0] for i in list(prefix_13_49['predict_act'])]

    anomal_anomal_acc = accuracy_score(y_true, y_pred)

    prefix_nap_acc[prefix] = [normal_normal_acc, normal_anomal_acc, 
                              anomal_normal_acc, anomal_anomal_acc, total_acc, anomal_f1]
prefix_nap_acc = pd.DataFrame(prefix_nap_acc).T
prefix_nap_acc = prefix_nap_acc.rename(columns={0:'N/N',1:'N/A',
                                                2:'A/N',3:'A/A',
                                                4:'Total',
                                                5:'Anomal_F1'})
prefix_nap_acc
# print(anomal_f1)

#%%
prefix_nap_acc = dict()
anomal_f1 =[]
for prefix in range(2,35):
    
    # result_49 = pd.read_csv('./result/0.049_noise.csv_classifier_xgb_50_random_sample.csv')
    # result_49 = pd.read_csv('./result/0.099_noise.csv_fixed_v2.csv')
    result_49 = pd.read_csv('./result/0.049_noise.csv_classifier_xgb_50_random_sample_napv2.csv')


    prefix_13_49 = result_49[result_49['prefix']==prefix]

    y_true = list(prefix_13_49['actual_act'])
    y_pred = [ast.literal_eval(i)[0] for i in list(prefix_13_49['predict_act'])]
    total_acc = accuracy_score(y_true, y_pred)
    anomal_f1 = classification_report(prefix_13_49['true_noise'], prefix_13_49['pred_ad_anom'],output_dict=True)['1']['f1-score']

    prefix_13_49 = prefix_13_49[prefix_13_49['true_noise']==0]
    prefix_13_49 = prefix_13_49[prefix_13_49['pred_ad_anom']==0]
    y_true = list(prefix_13_49['actual_act'])
    y_pred = [ast.literal_eval(i)[0] for i in list(prefix_13_49['predict_act'])]
    normal_normal_acc = accuracy_score(y_true, y_pred)
    prefix_13_49 = result_49[result_49['prefix']==prefix]
    prefix_13_49 = prefix_13_49[prefix_13_49['true_noise']==0]
    prefix_13_49 = prefix_13_49[prefix_13_49['pred_ad_anom']==1]
    y_true = list(prefix_13_49['actual_act'])
    y_pred = [ast.literal_eval(i)[0] for i in list(prefix_13_49['predict_act'])]
    normal_anomal_acc = accuracy_score(y_true, y_pred)

    prefix_13_49 = result_49[result_49['prefix']==prefix]
    prefix_13_49 = prefix_13_49[prefix_13_49['true_noise']==1]
    prefix_13_49 = prefix_13_49[prefix_13_49['pred_ad_anom']==0]
    y_true = list(prefix_13_49['actual_act'])
    y_pred = [ast.literal_eval(i)[0] for i in list(prefix_13_49['predict_act'])]

    anomal_normal_acc = accuracy_score(y_true, y_pred)

    prefix_13_49 = result_49[result_49['prefix']==prefix]
    prefix_13_49 = prefix_13_49[prefix_13_49['true_noise']==1]
    prefix_13_49 = prefix_13_49[prefix_13_49['pred_ad_anom']==1]
    y_true = list(prefix_13_49['actual_act'])
    y_pred = [ast.literal_eval(i)[0] for i in list(prefix_13_49['predict_act'])]

    anomal_anomal_acc = accuracy_score(y_true, y_pred)

    prefix_nap_acc[prefix] = [normal_normal_acc, normal_anomal_acc, 
                              anomal_normal_acc, anomal_anomal_acc, total_acc, anomal_f1]
prefix_nap_acc = pd.DataFrame(prefix_nap_acc).T
prefix_nap_acc = prefix_nap_acc.rename(columns={0:'N/N',1:'N/A',
                                                2:'A/N',3:'A/A',
                                                4:'Total',
                                                5:'Anomal_F1'})
prefix_nap_acc

#%%
result_99 = pd.read_csv('./result/0.099_noise.csv_fixed.csv')
raw_data = pd.read_csv('./data/0.099_noise.csv')
prefix_12 = result_99[result_99['prefix']==12]

nn_cases = prefix_12[prefix_12['true_noise']==0]
nn_cases = nn_cases[nn_cases['pred_noise']==0]
nn_cases = nn_cases['case_id']
dfn = raw_data[raw_data['Case ID'].isin(nn_cases)]
concating=[]
groups = dfn.groupby('Case ID')
for _, group in groups:
    group = group.reset_index(drop=True)
    group = group.iloc[:12, :]
    concating.append(group)
pd.concat(concating).to_csv('./0.099_prefix12_nn_cases_fixed.csv', index=False)

na_cases = prefix_12[prefix_12['true_noise']==0]
na_cases = na_cases[na_cases['pred_noise']==1]
na_cases = na_cases['case_id']
dfn = raw_data[raw_data['Case ID'].isin(na_cases)]
concating=[]
groups = dfn.groupby('Case ID')
for _, group in groups:
    group = group.reset_index(drop=True)
    group = group.iloc[:12, :]
    concating.append(group)
pd.concat(concating).to_csv('./0.099_prefix12_na_cases_fixed.csv', index=False)

an_cases = prefix_12[prefix_12['true_noise']==1]
an_cases = an_cases[an_cases['pred_noise']==0]
an_cases = an_cases['case_id']
dfn = raw_data[raw_data['Case ID'].isin(an_cases)]
concating=[]
groups = dfn.groupby('Case ID')
for _, group in groups:
    group = group.reset_index(drop=True)
    group = group.iloc[:12, :]
    concating.append(group)
pd.concat(concating).to_csv('./0.099_prefix12_an_cases_fixed.csv', index=False)

aa_cases = prefix_12[prefix_12['true_noise']==1]
aa_cases = aa_cases[aa_cases['pred_noise']==1]
aa_cases = aa_cases['case_id']
dfn = raw_data[raw_data['Case ID'].isin(aa_cases)]
concating=[]
groups = dfn.groupby('Case ID')
for _, group in groups:
    group = group.reset_index(drop=True)
    group = group.iloc[:12, :]
    concating.append(group)
pd.concat(concating).to_csv('./0.099_prefix12_aa_cases_fixed.csv', index=False)


#%%
dfn = pd.read_csv('./0.099_prefix12_nn_cases_fixed.csv')
dfn = dfn.sort_values(by=['Case ID', 'Timestamp'])
variants = (
    dfn
    .groupby('Case ID')['Activity']
    .apply(lambda seq: ' > '.join(seq))    # join with whatever separator you like
)
# 3) Count how many cases share each variant
variant_counts = variants.value_counts().reset_index()
variant_counts.columns = ['Variant','Case_Count']

# peek at the top few
print(variant_counts.tail())

dist = (variant_counts
        .groupby('Case_Count')
        .size()              # number of variants in each Case_Count
        .reset_index(name='Num_Variants')
        .sort_values('Case_Count')
       )

print(dist)

#%%
# --- 0. Settings -------------------------------------------------------------
CSV_PATH   = './0.099_prefix12_nn_cases_fixed.csv'
SEP        = ' > '          # how you join activities into a variant string
BUCKET_CUT = 5              # collapse everything >= this into one bucket

# --- 1. Load & prep ----------------------------------------------------------
dfn = pd.read_csv(CSV_PATH, parse_dates=['Timestamp'])
dfn = dfn.sort_values(['Case ID', 'Timestamp'])

# --- 2. Build one "variant" per case ----------------------------------------
variants = (dfn.groupby('Case ID')['Activity']
              .agg(SEP.join))                         # same as lambda seq: ' > '.join(seq)

# --- 3. Count cases per variant ---------------------------------------------
variant_counts = (variants.value_counts()
                           .rename_axis('Variant')
                           .reset_index(name='Case_Count'))

print("Cases per variant (tail):")
print(variant_counts.tail())

# --- 4. Bucket case-counts (1,2,3,4, ≥5) ------------------------------------
variant_counts['Bucket'] = np.where(
    variant_counts['Case_Count'] >= BUCKET_CUT,
    f'≥{BUCKET_CUT}',
    variant_counts['Case_Count'].astype(str)
)

# optional: enforce order
order = ['1', '2', '3', '4', f'≥{BUCKET_CUT}']
variant_counts['Bucket'] = pd.Categorical(variant_counts['Bucket'],
                                          categories=order,
                                          ordered=True)

# --- 5. Aggregate: how many variants & how many cases per bucket ------------
summary = (variant_counts
           .groupby('Bucket')
           .agg(Num_Variants=('Variant', 'size'),
                Total_Cases=('Case_Count', 'sum'))
           .reset_index()
           .sort_values('Bucket'))

print("\nDistribution of variants by case-count bucket:")
print(summary)

# --- 6. (Optional) Percentages & cumulative coverage ------------------------
summary['Pct_Variants'] = summary['Num_Variants'] / summary['Num_Variants'].sum()
summary['Cum_Variants'] = summary['Pct_Variants'].cumsum()

total_cases_overall = variant_counts['Case_Count'].sum()
summary['Pct_Cases'] = summary['Total_Cases'] / total_cases_overall
summary['Cum_Cases'] = summary['Pct_Cases'].cumsum()
summary['Pct_Cases'] = (summary['Total_Cases'] / total_cases_overall * 100) \
                           .round(2).astype(str) + '%'
print("\nWith percentages:")
print(summary)