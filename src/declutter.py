import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

# I define the stop_words here so I don't do it every time in the function below
# stop_words = stopwords.words('english')
# df = pd.read_csv("/Users/abdeas0a/Desktop/X2.csv", low_memory=False)
df = pd.read_csv("D:\Coding Projects\sigmod\X2.csv", low_memory=False)


def get_keywords(record, col):
    brand = record[col]
    if brand is None:
        brand = ""

    result = str(brand)
    result = result.lower()

    basic_punct = '-_,:;/()®™'  # extra symbols: ? ! \ * ~ {} [] © = *
    punct_to_space = str.maketrans(basic_punct, ' ' * len(basic_punct))  # map punctuation to space
    result = result.translate(punct_to_space)

    tokens = nltk.tokenize.word_tokenize(result)
    # keywords = [keyword for keyword in tokens if keyword.isalpha() and not keyword in stop_words]
    # keywords_string = ','.join(keywords)
    return tokens
    # return tokens[0]


def remove_bad_tokens(tokens):
    result_tokens = set()
    for el in tokens:
        if el in ['.', '']:
            continue
        result_tokens.add(el)
    return list(result_tokens)


# Turns each record into a list of tokens
def pre_process_record(record):
    idx = df_2.index.get_loc(record.name)

    for column in df_2.columns:
        tokens = remove_bad_tokens(get_keywords(record, column))
        df_2.iloc[idx][column] = tokens

    # all_tokens = tokens  # df_2.loc[df['brand']]
    # all_tokens = brand_tokens + cpu_brand_tokens
    # #return remove_bad_tokens(all_tokens)
    return


# Generate 'tokens' column. Doesn't work yet
def gather_tokens(cols):
    x = []
    for item in cols:
        x.append(item)
    return x


# axis=1 means that we will apply get_keywords to each row and not each column
df_2 = df.copy(deep=True)
df_2.apply(pre_process_record, axis=1)
df_2['tokens'] = df_2['brand'] + df_2['cpu_brand'] + df_2['cpu_model'] + df_2['cpu_type'] + df_2['cpu_frequency'] + \
                 df_2['ram_capacity'] + df_2['ram_type'] + df_2['ram_frequency'] + df_2['hdd_capacity'] + \
                 df_2['ssd_capacity'] + df_2['weight'] + df_2['dimensions']

df_2['tokens'] = df_2['tokens'].apply(remove_bad_tokens)
# df_2['tokens'] = df_2.apply(gather_tokens, axis=1)

print(df_2['title'].head(5))

x = 00

"""
regex = r'^(\d{1,2})\s?GB'
regex2=r'.?(\d{1,2}\.\d{1,2})\s?[lpk]'
#ram_c = df['ram_capacity'].str.extract(regex)
#weight= df['weight'].str.extract(regex2)
#ram_f=df['ram_frequency'].str.extract(r'^(\d{1})', expand=False)
#print(ram_c[35:40])
#ram_c.to_csv('modified3.csv', index=False, sep='\t')

#weight.to_csv('modified3.csv', index=False, sep='\t')
"""
