import pandas as pd
from sklearn.metrics import classification_report
df = pd.read_csv('./result/0.099_noise.csv_fixed.csv')

df12 = df[df['prefix'] == 12]
print(df.head())
true_noise = list(df12['true_noise'])
pred_noise = list(df12['pred_noise'])

true_noise.append(0)
pred_noise.append(1)

normal_normal = []
normal_anomal = []
anomal_nomal =  []
anomal_anomal = []


for pos, i in enumerate(true_noise):
    if i == 0 and pred_noise[pos] == 0:
        normal_normal.append(pos)
    elif i == 0 and pred_noise[pos] == 1:
        normal_anomal.append(pos)
    elif i == 1 and pred_noise[pos] == 0:
        anomal_nomal.append(pos)
    elif i == 1 and pred_noise[pos] == 1:
        anomal_anomal.append(pos)
        

print("N/N", len(normal_normal), "N/A", len(normal_anomal), "A/N", len(anomal_nomal), "A/A", len(anomal_anomal))
print(classification_report(y_true=true_noise, y_pred=pred_noise))
