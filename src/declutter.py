import csv
import pandas as pd
from operator import itemgetter
import math
import nltk
# from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

# I define the stop_words here so I don't do it every time in the function below
# stop_words = stopwords.words('english')
df = pd.read_csv("/Users/abdeas0a/Desktop/X2.csv", low_memory=False)


# This function will be applied to each row in our Pandas Dataframe
# See the docs for df.apply at:
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.apply.html
def get_keywords(record, col):
    # some_text = col['brand']
    brand = record[col]
    if brand is None:
        brand = ""

    result = str(brand)
    result = result.lower()

    # Remove ; , ,
    basic_punct = '?!,:;/\-~*_=(){}[]™' # add more œ∑´´†¥¨ˆˆπåß∂ƒ©˙∆˚¬Ω≈ç√∫˜˜≤≤‘“πøˆ¨¥æ¡™£¢∞§¶•ªº–ºª•πøˆ¨¥†®´´œåß∂ƒ©˙∆˚¬≥≤µ˜˜√ç≈…¬¥æ≥¡™£¢∞§¶•ª
    punct_to_space = str.maketrans(basic_punct, ' ' * len(basic_punct))  # map punctuation to space
    result = result.translate(punct_to_space)

    tokens = nltk.tokenize.word_tokenize(result)
    # keywords = [keyword for keyword in tokens if keyword.isalpha() and not keyword in stop_words]
    # keywords_string = ','.join(keywords)
    return tokens
    # return tokens[0]


def pre_process_record(record):
    # Update Brand
    brand_tokens = get_keywords(record, 'brand')
    df_2['brand'] = brand_tokens

    # Cpu brand
    cpu_brand_tokens = get_keywords(record, 'cpu_brand')
    df_2['cpu_brand'] = brand_tokens

    all_tokens = brand_tokens + cpu_brand_tokens

    result_tokens = set()
    for el in all_tokens:
        if el in ['.', '']:
            continue
        result_tokens.add(el)

    return list(result_tokens)


# applying the get_keywords function to our dataframe and saving the results
# as a new column in our dataframe called 'keywords'
# axis=1 means that we will apply get_keywords to each row and not each column
df_2 = df.copy(deep=True)
df_2['tokens'] = df_2.apply(pre_process_record, axis=1)
x = 00

"""
filtered_sent=[]
for w in tokenized_sent:
    if w not in stop_words:
        filtered_sent.append(w)
print("Tokenized Sentence:",tokenized_sent)
print("Filterd Sentence:",filtered_sent)

regex = r'^(\d{1,2})\s?GB'
regex2=r'.?(\d{1,2}\.\d{1,2})\s?[lpk]'
#ram_c = df['ram_capacity'].str.extract(regex)
#weight= df['weight'].str.extract(regex2)
#ram_f=df['ram_frequency'].str.extract(r'^(\d{1})', expand=False)
#print(ram_c[35:40])
#ram_c.to_csv('modified3.csv', index=False, sep='\t')

#weight.to_csv('modified3.csv', index=False, sep='\t')
"""

"""
def trim_to_null(c):
    return (
        f.lower(
            f.when(f.trim(f.col(c)) == '', None)
                .when(f.trim(f.col(c)) == 'null', None)
                .otherwise(f.trim(f.col(c)))
        )
    )


STRING_COLS = ['name', 'description', 'manufacturer']
for c in STRING_COLS:
    df = df.withColumn(c, f.lower(trim_to_null(c)))

STRING_NUM_COLS = ['price']
for c in STRING_NUM_COLS:
    df = df.withColumn(c, trim_to_null(c).cast('float'))


# hyphenated words and version numbers seems salient to product name
# treat them differently by concatenating
def replace_contiguous_special_char(c, replace_str=''):
    return (
        f.regexp_replace(c, "(?<=(\d|\w))(\.|-|\')(?=(\d|\w))", replace_str)
    )


def replace_special_char(c, replace_str=' '):
    return (
        f.regexp_replace(c, "[\W]", replace_str)
    )
"""
