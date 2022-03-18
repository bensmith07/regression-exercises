import itertools
import seaborn as sns
import matplotlib.pyplot as plt

def plot_variable_pairs(train):
    quant_features = [col for col in train.columns if train[col].dtype != object]
    feature_combos = list(itertools.combinations(quant_features, 2))
    for combo in feature_combos:
        sns.lmplot(x=combo[0], y=combo[1], data=train, line_kws={'color': 'red'})
        plt.show()
        
def months_to_years(df):
    df['tenure_years'] = df.tenure_months // 12
    return df

def plot_categorical_and_continuous_vars(train, categ_vars, cont_vars):    
    for cont_var in cont_vars:
        for categ_var in categ_vars:

            plt.figure(figsize=(30,10))

            plt.subplot(131)
            sns.barplot(data=train,
                        x=categ_var,
                        y=cont_var)
            plt.axhline(train[cont_var].mean(), 
                        ls='--', 
                        color='black')
            plt.title(f'{cont_var} by {categ_var}', fontsize=14)

            plt.subplot(132)
            sns.boxplot(data=train,
                          x=categ_var,
                          y=cont_var)

            sample_df = train.sample(1000)

            plt.subplot(133)
            sns.swarmplot(x=categ_var,
                          y=cont_var,
                          data=sample_df)

            plt.show()