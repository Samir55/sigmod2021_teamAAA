import pandas as pd
import os
import csv
import re
import logging
import optparse
import re
import spacy
import dedupe
import pickle
import copy
import json
from unidecode import unidecode


def formatNumber(num):
    num = float(num)
    if num % 1 == 0:
        return int(num)
    else:
        return num


def fill_nulls_with_none(df):
    """ Fills nulls in a dataframe with None.
        This is required for the Dedupe package to work properly.

        Input: - dataframe with nulls as NaN

        Output: - new dataframe with nulls as None
    """
    new_df = df.copy()
    for col in df.columns:
        new_df[col] = new_df[col].where(new_df[col].notnull(), None)
    return new_df


def convert_numbers_to_strings(df, cols_to_convert, remove_point_zero=True):
    """ Convert number types to strings in a dataframe.
        This is convoluted as need to keep NoneTypes as NoneTypes for what comes next!

        Inputs: - df -> dataframe to convert number types
                - cols_to_convert -> list of columns to convert
                - remove_point_zero -> bool to say whether you want '.0' removed from number

        Ouputs: - dataframe with converted number types
    """
    new_df = df.copy()
    for col in cols_to_convert:
        if remove_point_zero:
            new_df[col] = new_df[col].apply(lambda x: str(x).replace('.0', '') \
                if not isinstance(x, type(None)) else x)
        else:
            new_df[col] = new_df[col].apply(lambda x: str(x) \
                if not isinstance(x, type(None)) else x)
    return new_df


def clean_laptops_dataset(x_org):
    # Copy the dataset
    df = x_org.copy(deep=True)

    # Set the index
    df.set_index('instance_id', inplace=True, drop=False)

    sp = spacy.load('en_core_web_sm')

    # Read helper datasets stats
    extra_brands = set(pd.read_csv('../../data/sigmod/laptops.csv').Company.str.lower().unique())
    screen_sizes = set(pd.read_csv('../../data/sigmod/laptops.csv').Inches)
    screen_sizes = [str(formatNumber(str(s).lower())) for s in screen_sizes]

    # Keep only Alpha numeric
    irrelevant_regex = re.compile(r'[^a-z0-9,.\-\s]')
    multispace_regex = re.compile(r'\s\s+')  # Why it doesn't work
    df.replace({r'[^\x00-\x7F]+': ''}, regex=True, inplace=True)

    for column in df.columns:
        if column == 'instance_id':
            continue
        df[column] = df[column].str.lower().str.replace(irrelevant_regex, ' ').str.replace(multispace_regex, ' ')

    # Tokenize the new title
    def tokenize_new_tile(record):
        return [w.text for w in sp(record['new_title'])]

    df['new_title'] = df.title
    irrelevant_regex = re.compile(r'[^a-z0-9.\s]')
    multispace_regex = re.compile(r'\s\s+')  # TODO @Ahmed look at this
    df['new_title'] = df.new_title.str.lower().str.replace(irrelevant_regex, '').str.replace(multispace_regex, ' ')
    df['new_title_tokens'] = df.apply(tokenize_new_tile, axis=1)

    # Brand assignment
    all_brands = set(extra_brands)

    def assign_brand(record):
        # Search in brand first
        if record['brand'] in all_brands:
            return record['brand']
        # then in the title
        for el in all_brands:
            if el in record['title']:
                return el
        return "NNN"

    df['brand'] = df.apply(assign_brand, axis=1)

    # cpu brand
    def assign_cpu_brand(record):
        # Search in brand first
        if 'intel' in str(record['cpu_brand']) or 'intel' in str(record['title']) or \
                'intel' in str(record['cpu_model']) or 'intel' in str(record['cpu_type']):
            return 'intel'
        return 'amd'

    df['cpu_brand'] = df.apply(assign_cpu_brand, axis=1)

    def assign_screen_size(record):
        brand_tokens = record['new_title_tokens']
        arr = []
        for t in brand_tokens:
            s = t.replace('inch', '')
            s = s.replace('in', '')
            arr.append(s)

        for sc in screen_sizes:
            if str(sc) in arr:
                return str(sc)

        else:
            return str(15.6)  # Some relaxation

    df['screen_size'] = df.apply(assign_screen_size, axis=1)

    # ram capacity
    def assign_ram_capacity(record):
        s = str(record['ram_capacity']).replace(' ', '')
        possible_vals = ['2gb', '4gb', '6gb', '8gb', '10gb', '12gb', '16gb',
                         '32gb', '64gb', '128gb', '256gb', '512gb', '2', '4',
                         '6', '8', '10', '12', '16', '32', '64', '128']
        for val in possible_vals:
            if val in s:
                return int(val.replace('gb', ''))

        s = str(record['title']).replace(' ', '')  # This will be wrong, please change
        possible_vals = ['2gb', '4gb', '6gb', '8gb', '10gb', '12gb', '16gb',
                         '32gb', '64gb', '128gb']
        for val in possible_vals:
            if val in s:
                return int(val.replace('gb', ''))

        return 0

    def assign_hdd_capacity(record):
        s = str(record['hdd_capacity']).replace(' ', '')
        s2 = str(record['title'].replace(' ', ''))

        if 'ssd' in s:
            return 0

        if re.search("\d{3,4}gb", s):
            return int(re.findall("\d{3,4}gb", s)[0][:-2])
        if re.search("\dtb", s):
            return int(re.findall("\dtb", s)[0][:-2] + '000')
        if re.search("\d{3,4}gbhdd", s2):
            return int(re.findall("\d{3,4}gbhdd", s2)[0][:-5])
        if re.search("hdd\d{3,4}gb", s2):
            return int(re.findall("hdd\d{3,4}gb", s2)[0][3:-2])
        if re.search("hdd\d{1}tb", s2):
            return int(re.findall("hdd\d{1}tb", s2)[0][3:4] + '000')
        if re.search("\d{1}tbhdd", s2):
            return int(re.findall("\d{1}tbhdd", s2)[0][0] + '000')
        return 0

    df['hdd_capacity'] = df.apply(assign_hdd_capacity, axis=1)

    def assign_ssd_capacity(record):
        s = str(record['ssd_capacity']).replace(' ', '')
        s2 = str(record['title'].replace(' ', ''))

        if re.search("\d{3,4}gb", s):
            return int(re.findall("\d{3,4}gb", s)[0][:-2])
        if re.search("\dtb", s):
            return int(re.findall("\dtb", s)[0][:-2] + '000')
        if re.search("\d{3,4}gbssd", s2):
            return int(re.findall("\d{3,4}gbssd", s2)[0][:-5])
        if re.search("ssd\d{3,4}gb", s2):
            return int(re.findall("ssd\d{3,4}gb", s2)[0][3:-2])
        if re.search("ssd\d{1}tb", s2):
            return int(re.findall("ssd\d{1}tb", s2)[0][3:4] + '000')
        if re.search("\d{1}tbssd", s2):
            return int(re.findall("\d{1}tbssd", s2)[0][0] + '000')
        return 0

    df['ssd_capacity'] = df.apply(assign_ssd_capacity, axis=1)

    def assign_laptop_model(record):
        brand_tokens = record['new_title_tokens']
        try:
            brand_index = brand_tokens.index(str(record['brand']))
            finish_index = brand_index + 2
            should_break = False
            for i in range(2 + brand_index, 5 + brand_index, 1):
                for sc in screen_sizes:
                    if (sc in brand_tokens[i]):
                        should_break = True
                        break
                if should_break:
                    if finish_index == i:
                        finish_index -= 1
                    break
                if not (brand_tokens[i].isalpha()):
                    finish_index = i
                else:
                    break
        except:
            brand_index = -1

        if brand_index == -1:
            return None

        return ' '.join(brand_tokens[brand_index + 1:finish_index + 1])

    df['model'] = df.apply(assign_laptop_model, axis=1)
    df['ram_capacity'] = df.apply(assign_ram_capacity, axis=1)

    df = fill_nulls_with_none(df)
    df = convert_numbers_to_strings(df, ['screen_size'])

    # Unit stand. in weight TODO @Ahmed

    def assign_cpu_type(record):
        # Find the cpu type
        cpu_list = ["i5", "i3", "i7", "atom",
                    "pentium", "celeron", "a-series",
                    "e-series", "aseries", "eseries",
                    "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9"]

        for cpu in cpu_list:
            if record['cpu_type'] is not None and cpu in record['cpu_type']:
                return cpu
            if cpu in record['title']:
                return cpu
            if record['cpu_model'] is not None and cpu in record['cpu_model']:
                return cpu
            if record['cpu_frequency'] is not None and cpu in record['cpu_frequency']:
                return cpu

            if re.search("e-[0-9]{3}", record['title']):
                return re.findall("e-[0-9]{3}", record['title'])[0]

            if record['cpu_model'] is not None and re.search("e-[0-9]{3}", record['cpu_model']):
                return re.findall("e-[0-9]{3}", record['cpu_model'])[0]

    df['cpu_type'] = df.apply(assign_cpu_type, axis=1)

    return df


def clean_products_dataset(x_org):
    x4_dev = convert_numbers_to_strings(x_org, ['price']).copy(deep=True)
    x4_dev.set_index('instance_id', inplace=True)

    def get_type(record):
        name = record['name'].lower()

        if pd.isna(record['size']):
            if 'tv' in name:
                return 'tv'
            return 'mobile'

        flash_keywords = ['usb', 'drive']
        memory_stick_keywords = ['card', 'stick', 'sd', 'microsd', 'hc', 'class', 'speicherkarte']  # Add variants here

        is_flash = False
        is_memory = False

        for w in flash_keywords:
            if w in name:
                is_flash = True
                break

        for w in memory_stick_keywords:
            if w in name:
                is_memory = True
                break

        if is_flash:
            return 'flash'

        if is_memory:
            return 'stick'

        return 'stick'

    with open('../../data/sigmod/translations_lookup_all.json') as fin:
        variants = json.load(fin)

    with open('../../data/sigmod/langs_dict.json') as fin:
        json.load(fin)

    # Alpha numeric
    irrelevant_regex = re.compile(r'[^a-z0-9,.\-\s]')
    multispace_regex = re.compile(r'\s\s+')  # Why it doesn't work
    x4_dev.replace({r'[^\x00-\x7F]+': ''}, regex=True, inplace=True)

    for column in x4_dev.columns:
        if column == 'instance_id':
            continue
        x4_dev[column] = x4_dev[column].str.lower().str.replace(irrelevant_regex, ' ').str.replace(multispace_regex,
                                                                                                   ' ')

    x4_dev['product_type'] = x4_dev.apply(get_type, axis=1)
    x4_dev.drop('price', inplace=True, axis=1)
    x4_dev['size'] = x4_dev['size'].str.lower().str.replace(' ', '')
    x4_dev['size'] = x4_dev['size'].where(x4_dev['size'].notnull(), 0)

    # Remove unwanted words from the name
    for i in range(len(x4_dev)):
        record = x4_dev.iloc[i]

        name = record['name']

        # remove unnecessary characters
        basic_punct = '-/\*_,:;/()®™'
        punct_to_space = str.maketrans(basic_punct, ' ' * len(basic_punct))  # map punctuation to space
        name = name.translate(punct_to_space)

        # remove brand
        name = name.replace(record['brand'], '')

        # remove size

        if record.product_type in ['flash', 'stick']:
            name = re.sub('\d\d\d\s?gb', '', name, 6)
            name = re.sub('\d\d\s?gb', '', name, 6)
            name = re.sub('\d\s?gb', '', name, 6)

        tokens = name.split(' ')
        for wd, wdtl in variants.items():
            while wd in tokens:
                tokens.remove(wd)
            for wdt in wdtl:
                while wdt in tokens:
                    tokens.remove(wdt)

        unneeded_words = ['mmoire', 'speicherkarte', 'flashgeheugenkaart', 'flash', 'stick', 'speed', 'high']
        for w in unneeded_words:
            while w in tokens:
                tokens.remove(w)
        x4_dev.iloc[i]['name'] = ' '.join(tokens)

    for column in x4_dev.columns:
        if column == 'instance_id':
            continue
        x4_dev[column] = x4_dev[column].str.lower().str.replace(irrelevant_regex, ' ').str.replace(multispace_regex,
                                                                                                   ' ')

    return x4_dev
