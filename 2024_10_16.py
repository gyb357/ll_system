# %%
import pandas as pd
import numpy as np
# %%
# 하와이 코로나 데이터 전처리
raw_hawaii = pd.read_csv('dataset/hawii_covid.csv')
# print(raw_hawaii.head())

hawaii_data = raw_hawaii[['date_updated', 'tot_cases']]
# print(hawaii_data.head())

hawaii_data_idx = hawaii_data.set_index('date_updated')
# print(hawaii_data_idx.head())

hawaii_data_df = hawaii_data_idx['tot_cases']
print(hawaii_data_df.head())
# %%
# 한국 코로나 데이터 전처리
raw_world = pd.read_csv('dataset/owid-covid-data.csv')
# print(raw_world.head())
# print(raw_world['location'].unique())

raw_date = raw_world[['date', 'total_cases', 'location']]
korea_data = raw_date[raw_date['location'] == 'South Korea']
# print(korea_data.head())

korea_data_idx = korea_data.set_index('date')
# print(korea_data_idx.head())

korea_data_df = korea_data_idx['total_cases']
print(korea_data_df.head())
# %%
# datetime 형식으로 변환
hawaii_data_idx.index = pd.to_datetime(hawaii_data_idx.index, format='%m/%d/%Y')
korea_data_idx.index = pd.to_datetime(korea_data_idx.index, format='%Y-%m-%d')
# print(hawaii_data_df.head())
# print(korea_data_df.head())

korea_data_filtered = korea_data_idx[korea_data_idx.index.isin(hawaii_data_idx.index)]

# Merge the two datasets
final_df = pd.DataFrame(
    {
    'hawaii': hawaii_data_idx['tot_cases'],
    'korea': korea_data_filtered['total_cases']
    },
    index=hawaii_data_idx.index
)
print(final_df.head())
# %%
final_df.plot.line(rot=45)
# %%
df = pd.read_csv('dataset/survey_results_public.csv')
# print(df.head())

for col in df.columns:
    # print(col)
    pass

reversed_df = df[
    [
        'Age',
        'Country',
        'LearnCode',
        'LanguageHaveWorkedWith',
        'LanguageWantToWorkWith'
    ]
]
# print(reversed_df.head())
# print(reversed_df['Age'].duplicated().head())

size_by_age = reversed_df.groupby('Age').size()
print(size_by_age.head())
# size_by_age.plot.bar()

reindexed_age = size_by_age.reindex(
    index=(
        'Under 18 years old',
        '18-24 years old',
        '25-34 years old',
        '35-44 years old',
        '45-54 years old',
        '55-64 years old',
        '65 years or older'
    )
)
print(reindexed_age.head())
# %%
reindexed_age.plot.bar()
# %%
reindexed_age.plot.barh()
# %%
reindexed_age.plot.pie()
# %%
size_by_contry = reversed_df.groupby('Country').size()
# print(size_by_contry.head())

# 상위 20개 국가
# top_10_country = size_by_contry.nlargest(20).plot.pie()

languages = reversed_df["LanguageHaveWorkedWith"].str.split(";", expand=True)
# print(languages.head())

size_by_language = languages.stack().value_counts()
# print(size_by_language.head())

# %%
size_by_language.nlargest(10).plot.bar()
# %%
size_by_language.nlargest(10).plot.pie()
# %%
