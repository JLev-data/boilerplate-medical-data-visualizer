import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] =  np.where(df['weight'] / (df['height']/100)**2 > 25,1,0)

# 3
for col in ['cholesterol','gluc']:
    df[col] = np.where(df[col] == 1 ,0,1)

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df,id_vars=['id','cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6
    df_cat = df_cat.groupby(['cardio','variable','value']).count()
    df_cat.rename(columns={'id':'total'},inplace=True)

    # 7
    df_cat = df_cat.reset_index()
    graph = sns.catplot(df_cat,x='variable', y='total', hue='value', col='cardio', kind='bar')

    # 8
    fig = graph.fig

    # 9
    fig.savefig('catplot.png')
    return fig

# 10
def draw_heat_map():
    # 11 
    df_heat = df.loc[ (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) & (df['height'] <= df['height'].quantile(0.975)) & 
        (df['weight'] <= df['weight'].quantile(0.975)) & (df['weight'] >= df['weight'].quantile(0.025)) ]

   # 12
    corr = df_heat.corr()

    # 13
    mask = np.zeros(corr.shape)
    for i  in range(corr.shape[0]):
        for j  in range(corr.shape[1]):
            if i <= j :
                mask[i,j] = 1

    # 14
    fig, ax = plt.subplots()

    # 15
    sns.heatmap(corr,mask=mask, cmap='icefire', vmax=0.24, vmin=-0.16, annot=True, fmt='.1f', 
            annot_kws={"size": 5}, cbar_kws={"shrink": .5}, linewidths=.1,square=True)
    plt.tight_layout()
    # 16
    fig.savefig('heatmap.png')
    return fig


