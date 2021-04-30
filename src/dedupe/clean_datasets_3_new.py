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

    spacy.cli.download("en_core_web_sm")  # TODO uncomment
    sp = spacy.load('en_core_web_sm')

    # Read helper datasets stats
    extra_brands = {'acer', 'google', 'toshiba', 'dell', 'xiaomi', 'asus', 'mediacom', 'hp', 'vero', 'lg',
                    'chuwi', 'lenovo', 'apple', 'microsoft', 'fujitsu', 'huawei', 'samsung', 'razer', 'msi'}
    screen_sizes = {'14.1', '15.6', '12.5', '12', '15', '13.5', '14', '17', '17.3',
                    '12.3', '13.9', '15.4', '10.1', '13', '11.6', '18.4', '13.3', '11.3'}
    # extra_brands = set(
    #    pd.read_csv('../../data/sigmod/laptops.csv', encoding='windows-1251').Company.str.lower().unique())
    # screen_sizes = set(pd.read_csv('../../data/sigmod/laptops.csv', encoding='windows-1251').Inches)
    # screen_sizes = [str(formatNumber(str(s).lower())) for s in screen_sizes]

    # Keep only Alpha numeric
    irrelevant_regex = re.compile(r'[^a-z0-9,.\-\s]')
    multispace_regex = re.compile(r'\s\s+')  # Why it doesn't work
    multispace_regex_2 = re.compile(r'\s-\s')  # Why it doesn't work
    df.replace({r'[^\x00-\x7F]+': ''}, regex=True, inplace=True)

    for column in df.columns:
        if column == 'instance_id':
            continue
        df[column] = df[column].str.lower().str.replace(irrelevant_regex, ' ').str.replace(multispace_regex, ' ')

    df['title_org'] = df['title']

    # Tokenize the new title
    def tokenize_new_tile(record):
        return [w.text for w in sp(record['new_title'])]

    remove_words = ['with', 'clarinet', 'audiophile', 'end', 'pc', 'french', 'performance', '"', 'burner', 'sd',
                    'canada', 'certified',
                    'keyboard', 'backlight', 'professional', 'at', 'beats', 'drive', 'microphone', 'vology',
                    'america',
                    'refurbished', 'computer', 'dimm', 'ultrabase', 'audio', ':', 'switching', 'premium', 'special',
                    'dvd', 'portable',
                    'speaker', 'buy.net', 'downgrade', '/', '&', 'wireless', 'home', 'notebook', ')', 'edition',
                    'built-in',
                    'dualcore', 'high', 'revolve', 'cool', 'and', 'micro', 'aluminum', 'tigerdirect', 'voice',
                    'nx.m8eaa.007',
                    'comfyview', 'amazon.com', 'bes', 'ultraportable', 'gb', 'core', 'computers', 'screen', 'slot',
                    'lan', 'supermulti', 'technology', 'bluray', 'price', 'display', 'dvdrw', '.com',
                    'internationalaccessories',
                    'touch', 'card', 'us', 'bluetooth', 'dvdwriter', 'for', 'new', 'comparison', 'webcam', '(',
                    'laptop',
                    'accessories', 'brand', 'builtin', 'dvd', 'batt', 'walmart.com', 'ebay', 'gaming', 'windows',
                    'laptop', 'sealed', 'wifi', 'best', 'topendelectronics', 'cd', 'win']  # , 'g']

    replace_words = {'hewlett-packard': 'hp'}

    def clean_title(record):
        title = record['title']

        # Remove unneeded words
        for w in remove_words:
            title = title.replace(w, '')

        # Replace words with common word
        for w, fix_w in replace_words.items():
            title = title.replace(w, fix_w)

        return title

    df['new_title'] = df.apply(clean_title, axis=1)
    irrelevant_regex = re.compile(r'[^a-z0-9.\s]')
    multispace_regex = re.compile(r'\s\s+')
    df['new_title'] = df.new_title.str.lower().str.replace(irrelevant_regex, '').str.replace(multispace_regex,
                                                                                             ' ').str.replace(
        multispace_regex_2, ' ')
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
        return None

    df['brand'] = df.apply(assign_brand, axis=1)

    # cpu brand
    intel = ['intel', 'i3', 'i5', 'i7', 'celeron', 'pentium']  # Needed because not all entries have intel

    def assign_cpu_brand(record):
        # Search in brand first
        for blue in intel:
            if blue in str(record['cpu_brand']) or blue in str(record['title']) or \
                    blue in str(record['cpu_model']) or blue in str(record['cpu_type']):
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
            return None

    df['screen_size'] = df.apply(assign_screen_size, axis=1)

    # ram capacity
    def assign_ram_capacity(record):
        s = str(record['ram_capacity'])
        t = str(record['title'])
        regex = re.compile(r'(\d{1,3})\s?([gm]b)')  # rare chance of encountering MB as an error
        m = None
        if s:
            m = re.search(regex, s)
        if m is None:
            m = re.search(regex, t)

        if m is None:
            return None

        m = m.group()
        res = (re.sub(r'([gm]b)', "gb", m)).replace(' ', '')

        ram = int(res[:-2])

        if ram > 64:
            return None

        return res

    def assign_hdd_capacity(record):
        s = str(record['hdd_capacity']).replace(' ', '')
        s2 = str(record['title'].replace(' ', ''))

        if 'ssd' in s:
            return 0

        res = None

        if re.search("\d{3,4}gb", s):
            res = str(re.findall("\d{3,4}gb", s)[0][:-2]) + ' gb'
        elif re.search("\dtb", s):
            res = str(re.findall("\dtb", s)[0][:-2] + '000') + ' gb'
        elif re.search("\d{3,4}gbhdd", s2):
            res = str(re.findall("\d{3,4}gbhdd", s2)[0][:-5]) + ' gb'
        elif re.search("hdd\d{3,4}gb", s2):
            res = str(re.findall("hdd\d{3,4}gb", s2)[0][3:-2]) + ' gb'
        elif re.search("hdd\d{1}tb", s2):
            res = str(re.findall("hdd\d{1}tb", s2)[0][3:4] + '000') + ' gb'
        elif re.search("\d{1}tbhdd", s2):
            res = str(re.findall("\d{1}tbhdd", s2)[0][0] + '000') + ' gb'

        if res is None:
            return None

        return res

    df['hdd_capacity'] = df.apply(assign_hdd_capacity, axis=1)

    def assign_ssd_capacity(record):
        s = str(record['ssd_capacity']).replace(' ', '')
        s2 = str(record['title'].replace(' ', ''))

        if re.search("\d{3}gbssd", s):
            return str(re.findall("\d{3}gb", s)[0][:-2]) + ' gb'
        if re.search("\dtbssd", s):
            return str(re.findall("\dtb", s)[0][:-2] + '000') + ' gb'
        if re.search("\d{3}gbssd", s2):
            return str(re.findall("\d{3}gbssd", s2)[0][:-5]) + ' gb'
        if re.search("ssd\d{3}gb", s2):
            return str(re.findall("ssd\d{3}gb", s2)[0][3:-2]) + ' gb'
        if re.search("ssd\d{1}tb", s2):
            return str(re.findall("ssd\d{1}tb", s2)[0][3:4] + '000') + ' gb'
        if re.search("\d{1}tbssd", s2):
            return str(re.findall("\d{1}tbssd", s2)[0][0] + '000') + ' gb'
        return None

        # if re.search("\d{3,4}gbssd", s):
        #     return str(re.findall("\d{3,4}gb", s)[0][:-2]) + ' gb'
        # if re.search("\dtbssd", s):
        #     return str(re.findall("\dtb", s)[0][:-2] + '000') + ' gb'
        # if re.search("\d{3,4}gbssd", s2):
        #     return str(re.findall("\d{3,4}gbssd", s2)[0][:-5]) + ' gb'
        # if re.search("ssd\d{3,4}gb", s2):
        #     return str(re.findall("ssd\d{3,4}gb", s2)[0][3:-2]) + ' gb'
        # if re.search("ssd\d{1}tb", s2):
        #     return str(re.findall("ssd\d{1}tb", s2)[0][3:4] + '000') + ' gb'
        # if re.search("\d{1}tbssd", s2):
        #     return str(re.findall("\d{1}tbssd", s2)[0][0] + '000') + ' gb'
        # return None

    def has_ssd(record):
        s = str(record['ssd_capacity']).replace(' ', '')
        s2 = str(record['title'].replace(' ', ''))

        return 'ssd' in s + ' ' + s2

    df['ssd_capacity'] = df.apply(assign_ssd_capacity, axis=1)
    df['has_ssd'] = df.apply(has_ssd, axis=1)

    def assign_laptop_model(record):
        brand = record['brand']
        t = record['new_title']

        if brand == 'acer':
            regex = [r'\s[esrvmpa]\d-?\s?.....-?....?', r'\s[esrvma]?\d-....', r'\s?acer\saspire\s\d{4}-\d{4}',
                     r'\s?acer\sextensa\s.{0,2}\d{4}\s?\d{0,4}', r'\s?acer\saspire\sas\d{4}-?\s?\d{4}',
                     r'\s?acer\saspire\s\d{4}-?\s\d{4}']
            for r in regex:
                cr = re.compile(r)
                if re.search(cr, t):
                    res = re.search(cr, t).group()
                    for w in ['extensa', 'acer', 'aspire']:
                        res = res.replace(w, '')
                    return res.lstrip().rstrip()

        if brand == 'asus':
            regex = [r'\sux...-.....', r'\sux.{3,5}-?.....?']  # There is a problem here TODO @Ahmed
            for r in regex:
                cr = re.compile(r)
                if re.search(cr, t):
                    return re.search(cr, t).group().lstrip().rstrip()

        if brand == 'lenovo':
            regex = [r'\sx\d{3}\s?tablet?\s?\d{0,4}', r'\sx\d{3}\s?laptop?\s?\d{0,4}', r'\sx\d{3}\s?\d{0,4}',
                     r'\sx\d{1}\scarbon\s\d{4}', r'\sx\d{1}\scarbon touch\s\d{4}']

            for r in regex:
                cr = re.compile(r)
                if re.search(cr, t):
                    res = re.search(cr, t).group()
                    for w in ['carbon', 'touch', 'tablet', 'laptop']:
                        res = res.replace(w, '')
                    return res.lstrip().rstrip()

        if brand == 'hp':
            regex = [r'\sfolio\s?\d{4}.', r'\selitebook\s?-?\d{3,4}.',
                     r'\s\d{2}-.{4,6}', r'hp\s?\d{4}.', r'\spavilion\s?..\s?.{5}',
                     r'\s?compaq\s?.{5,6}', r'\s?hp\s?15\s?[a-z]\d{3}[a-z]{1,2}', r'\shp\s?12\s?.{5}',
                     r'\s?elitebook\srevolve\s?\d{3}\s?', '\s.\d{3}[pgmwm][pgmwm]?']
            for r in regex:
                cr = re.compile(r)
                if re.search(cr, t):
                    res = re.search(cr, t).group()
                    for w in ['folio', 'elitebook', 'hp', 'compaq', 'pavilion', ' 15 ', ' 12 ', 'revolve']:
                        res = res.replace(w, '')
                    return res.lstrip().rstrip()

        if brand == 'dell':
            regex = [r'\s[nmi]\d{3,4}(-\d{4})?', r'\sinspiron\s15?\s?\d{4}', r'\sinspiron\s17?.?\s?\d{4}',
                     r'\slatitude\s15?\s?\d{4}']
            for r in regex:
                cr = re.compile(r)
                if re.search(cr, t):
                    res = re.search(cr, t).group()
                    for w in ['inspiron', '15', '17r', '   ', '  ', 'latitude']:
                        res = res.replace(w, '')
                    return res.lstrip().rstrip()

        return None

    df['model'] = df.apply(assign_laptop_model, axis=1)
    df['ram_capacity'] = df.apply(assign_ram_capacity, axis=1)

    df = fill_nulls_with_none(df)
    df = convert_numbers_to_strings(df, ['screen_size'])

    def assign_model_name(record):  # laptop Line
        # print(record['model'].split())
        if record['model'] is None:
            return None;
        ans = record['model'].split(" ")[0]
        if ans.isalpha():
            return ans
        return None

    df['model_name'] = df.apply(assign_model_name, axis=1)

    def assign_cpu_model(record):
        model = record['cpu_model']
        if record['cpu_type'] is not None:
            if model is not None:
                model += ' '
                model += record['cpu_type']
            else:
                model = record['cpu_type']

        regex = re.compile(r"-?\d{3,4}([mul]{1,2})")  # For intel cpus
        regex2 = re.compile(r"[ea]\d?-\d{1,4}[m]?")  # for amd A and E series. Needs detection after AMD tag in title
        m = None
        if record['cpu_brand'] == 'intel' and model is not None:
            m = re.search(regex, model)
            if m is not None:
                m = m.group()
                return re.sub(r'-', "", m)
            #
            # regex = re.compile(r"i[3579][-\s]?\d{3,4}.")  # For intel cpus
            # m = re.search(regex, model)
            # if m is not None:
            #     m = m.group()
            #     return re.sub(r'-', "", m)

        if re.search("intel", record['title']):  # one case where laptop model is 50m and gets caught
            m = re.search(regex, record['title'])
            if m is not None:
                m = m.group()
                return re.sub(r'-', "", m)

        if record['cpu_brand'] == 'amd' and model is not None:
            m = re.search(regex2, model)
            if m is not None:
                m = m.group()
                return re.sub(r'[ea]\d?-', "", m)
        if re.search("amd", record['title']):
            m = re.search(regex2, record['title'])
            if m is not None:
                m = m.group()
                return re.sub(r'[ea]\d?-', "", m)
        if m is None:
            return None

    df['cpu_model'] = df.apply(assign_cpu_model, axis=1)

    def assign_cpu_type(record):
        # Find the cpu type
        cpu_list = ["i5", "i3", "i7", "atom",
                    "pentium", "celeron", "a-series",
                    "e-series", "aseries", "eseries",
                    "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9"]

        for cpu in cpu_list:
            if record['cpu_type'] is not None and cpu in str(record['cpu_type']):
                return cpu

            if record['cpu_model'] is not None and cpu in str(record['cpu_model']):
                return cpu
            if record['cpu_frequency'] is not None and cpu in str(record['cpu_frequency']):
                return cpu

            if cpu in str(record['title']):
                return cpu

            if re.search("e-[0-9]{3}", record['title']):
                return re.findall("e-[0-9]{3}", record['title'])[0]

            if record['cpu_model'] is not None and re.search("e-[0-9]{3}", record['cpu_model']):
                return re.findall("e-[0-9]{3}", record['cpu_model'])[0]

        return None

    df['cpu_type'] = df.apply(assign_cpu_type, axis=1)

    def assign_cpu_frequency(record):
        s = record['cpu_frequency']
        regex = re.compile(r"\d?.\d{1,2}\s?ghz")
        m = None
        if s:
            m = re.search(regex, s)
            if m is not None:
                m = m.group()
                return re.sub(r'ghz', "", m)
        if re.search("ghz", record['title']):
            m = re.search(regex, record['title'])
            if m is not None:
                m = m.group()
                return re.sub(r'ghz', "", m)
        if m is None:
            return None

    df['cpu_frequency'] = df.apply(assign_cpu_frequency, axis=1)

    def assign_new_title(record):
        # Remove extracted data from the title

        # Remove model name
        record['new_title'] = record['nwe_titl']

        # Remove brand
        # Remove screen size
        # Remove cpu brand
        # Remove cpu type
        # Ram capacity, hdd capacity, ssd capacity

    def assign_cpu(record):
        cpu_type = record['cpu_type']
        cpu_model = record['cpu_model']

        res = ""
        if cpu_type is not None:
            res += cpu_type

        if cpu_model is not None:
            res += '-'
            res += cpu_model

        return res

    df['cpu_model'] = df.apply(assign_cpu, axis=1)

    df['new_title_2'] = df['new_title']

    return df
