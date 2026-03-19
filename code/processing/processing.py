import pandas as pd
import os
import tsv_cleaner

# clean the tsv files as necessary

tsv_cleaner.clean_tsv("../data/raw/Organizations.tsv", "../data/clean/Organizations.tsv")

# read each tsv file into a pandas df

df_organizations = pd.read_csv(
    '../data/clean/Organizations.tsv', 
    sep='\t', 
    names=[
        'oid',
        'name',
        'city',
        'state',
        'type',
        "lastTouched", 
        "analysis_flag",
        'lastTouched_ts',
        'source',
        'verified',
        'oc_id',
        'org_type',
    ],
    engine='python')

df_categories = pd.read_csv(
    '../data/clean/OrgCategories.tsv', 
    sep='\t', 
    names=[
        'oid',
        'category_level',
        'category',
        'source',
        'lastTouched'
    ],
    engine='python')

# weird fields are denoted by ghost --> LOOK INTO
df_concepts = pd.read_csv(
    '../data/clean/OrgConcept.tsv', 
    sep='\t', 
    names=[
        'oc_id',
        'name',
        'acronym',
        'canon_oid',
        'state',
        'ghost1',
        'validated',
        'ghost2',
        'merged_into',
        'to_be_deleted',
        'lastTouched',
        'ghost3'
    ],
    engine='python')

# drop unneccesary/useless columns from each df

df_organizations = df_organizations.drop(columns=["city", 
                                            "state", 
                                            "type", 
                                            "lastTouched", 
                                            "analysis_flag", 
                                            "lastTouched_ts", 
                                            "source",
                                            "verified",
                                            "org_type"])

df_categories = df_categories.drop(columns=["lastTouched", "source"])

df_concepts = df_concepts.drop(columns=['state',
                                        'ghost1',
                                        'validated',
                                        'ghost2',
                                        'merged_into',
                                        'to_be_deleted',
                                        'lastTouched',
                                        'ghost3'])

# pivot categories to long format

df_cat_long = df_categories.pivot_table(
    index='oid',
    columns='category_level',
    values='category',
    aggfunc=lambda x: x.iloc[0]  # take the first value if duplicates exist (gets around uncoded duplicate rows)
).reset_index()

# ensure the values we join on have consistent typing to avoid errors

df_organizations['oid'] = df_organizations['oid'].astype('Int64')
df_cat_long['oid'] = df_cat_long['oid'].astype('Int64')
df_organizations['oc_id'] = df_organizations['oc_id'].astype('Int64')
df_concepts['oc_id'] = df_concepts['oc_id'].astype('Int64')

# inner join orgs and categories on oid

df_org_categories = df_organizations.merge(df_cat_long, on="oid", how="inner")

# join the categories into the result of the previous join

df_full = df_org_categories.merge(df_concepts, on='oc_id', how='left', suffixes=('_org', '_concept'))

# write results to file

df_full.to_csv("../data/processed/OrganizationsFull.tsv", sep="\t")
