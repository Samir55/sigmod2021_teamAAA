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


def clean_products_dataset_old(x_org):
    spacy.cli.download("en_core_web_sm")

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


def clean_products_dataset(x_org):
    replace_words = {
        'professional': 'pro',
        'data traveler': 'datatraveler',
        ' hx ': 'hyperx',
        'generation': 'g',
        ' micro sd': ' microsd',
        ' extrem ': 'extreme',
        ' classe ': 'class'
    }

    remove_words = ['tesco', 'direct', 'accessoires', 'montres', 'bracelets', 'connects']

    all_brands = ['lexar', 'sony', 'sandisk', 'pny', 'kingston', 'samsung',
                  'intenso', 'toshiba', 'transcend']

    brand_lines = {'lexar': {'stick': ['xqd', 'xqd pro', 'platinum'],
                             'flash': ['jumpdrive'],
                             'mobile': [],
                             'tv': []},
                   'sony': {'stick': [], 'flash': [], 'mobile': [], 'tv': []},
                   'sandisk': {'stick': ['extreme pro', 'ultra plus', 'ultra'],
                               'flash': ['cruzer glide', 'cruzer edge', 'cruzer fit', 'cruzer'],
                               'mobile': [], 'tv': []},
                   'pny': {'stick': [], 'flash': [], 'mobile': [], 'tv': []},
                   'kingston':
                       {
                           'stick': ['ultimate', 'hyperx'],
                           'flash': ['datatraveler', 'hyperx savage', 'hyperx'],
                           'mobile': [], 'tv': []
                       },
                   'samsung': {'stick': [], 'flash': [], 'mobile': [], 'tv': []},
                   'intenso': {'stick': [], 'flash': [], 'mobile': [], 'tv': []},
                   'toshiba': {'stick': ['exceria pro', 'exceria'],
                               'flash': [], 'mobile': [], 'tv': []},
                   'transcend': {'stick': [], 'flash': [], 'mobile': [], 'tv': []}}

    model_regex = {'lexar': {
        'stick': [r'\s\d{3,4}x'],
        'flash': [r'\sv\d\d.?', r'\sp\d\d.?', r'\sc\d\d.?', r'\ss\d\d.?'],
        'mobile': [],
        'tv': []
    },
        'sony': {'stick': [], 'flash': [], 'mobile': [], 'tv': []},
        'sandisk': {'stick': [r'\s\d{4}x', r'x\d{2}', ],
                    'flash': [], 'mobile': [], 'tv': []},
        'pny': {'stick': [], 'flash': [], 'mobile': [], 'tv': []},

        'kingston': {'stick': [],
                     'flash': [r'dt\d\d\d(g\d)?', r'\sg\d', r'101', r'g\d'],
                     'mobile': [], 'tv': []},

        'samsung': {'stick': [], 'flash': [], 'mobile': [], 'tv': []},
        'intenso': {'stick': [], 'flash': [], 'mobile': [], 'tv': []},
        'toshiba': {'stick': [r'[umn]\d{2,4}.?'], 'flash': [r'[umn]\d{2,4}.?'], 'mobile': [], 'tv': []},
        'transcend': {'stick': [], 'flash': [], 'mobile': [], 'tv': []}}

    class_regex = [r'class\s?\d{1,2}\s', r'\sc\d{1,2}\s']

    spacy.cli.download("en_core_web_sm")

    x4_dev = convert_numbers_to_strings(x_org, ['price']).copy(deep=True)
    x4_dev.set_index('instance_id', inplace=True)

    def clean_name(record):
        title = record['name']

        # Remove unneeded words
        for w in remove_words:
            title = title.replace(w, '')

        # Replace words with common word
        for w, fix_w in replace_words.items():
            title = title.replace(w, fix_w)

        return title

    x4_dev['name'] = x4_dev.apply(clean_name, axis=1)

    def get_type(record):
        name = record['name'].lower()

        for b, d in brand_lines.items():
            for w in d['stick']:
                if w in name:
                    return 'stick'

            for w in d['flash']:
                if w in name:
                    return 'flash'

        if pd.isna(record['size']):
            if 'tv' in name:
                return 'tv'
            return 'mobile'

        flash_keywords = ['usb', 'drive', 'flashdisk', 'cruzer']
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
    x4_dev['name_2'] = x4_dev.name

    def get_line(record):
        name = record['name'].lower()
        brand = record['brand']
        product_type = record['product_type']

        for w in brand_lines[brand][product_type]:
            if w in name:
                return w

        return None

    def get_model_number(record):
        name = record['name'].lower()
        brand = record['brand']
        product_type = record['product_type']

        for t in model_regex[brand][product_type]:
            cr = re.compile(t)
            if re.search(cr, name):
                return re.search(cr, name).group()

        return None

    x4_dev['line'] = x4_dev.apply(get_line, axis=1)
    x4_dev['model'] = x4_dev.apply(get_model_number, axis=1)

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
