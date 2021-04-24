
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


sp = spacy.load('en_core_web_sm')

def format_number(num):
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

        Outputs: - dataframe with converted number types
    """
    new_df = df.copy()
    for col in cols_to_convert:
        if remove_point_zero:
            new_df[col] = new_df[col].apply(lambda x: str(x).replace('.0','')\
                                            if not isinstance(x, type(None)) else x)
        else:
            new_df[col] = new_df[col].apply(lambda x: str(x)\
                                            if not isinstance(x, type(None)) else x)
    return new_df

extra_brands = {'acer', 'google', 'toshiba', 'dell', 'xiaomi', 'asus', 'mediacom', 'hp', 'vero', 'lg',
                    'chuwi', 'lenovo', 'apple', 'microsoft', 'fujitsu', 'huawei', 'samsung', 'razer', 'msi'}
screen_sizes = {'11.3', '14.1', '10.1', '12', '11.6', '13.3', '14', '15.4', '15.6', '17.3', '13.5',
                '12.5', '13', '18.4', '13.9', '17', '15', '12.3'}


def clean_laptops_dataset(x_org):
    # Copy the dataset
    df = x_org.copy(deep=True)

    # Set the index
    df.set_index('instance_id', inplace=True, drop=False)

    #spacy.cli.download("en_core_web_sm")
    sp = spacy.load('en_core_web_sm')

    # Read helper datasets stats
    extra_brands = {'acer', 'google', 'toshiba', 'dell', 'xiaomi', 'asus', 'mediacom', 'hp', 'vero', 'lg',
                    'chuwi', 'lenovo', 'apple', 'microsoft', 'fujitsu', 'huawei', 'samsung', 'razer', 'msi'}
    screen_sizes = {'14.1', '15.6', '12.5', '12', '15', '13.5', '14', '17', '17.3',
                    '12.3', '13.9', '15.4', '10.1', '13', '11.6', '18.4', '13.3', '11.3'}

    # Keep only Alpha numeric
    irrelevant_regex = re.compile(r'[^a-z0-9,.\-\s]')
    multispace_regex = re.compile(r'\s\s+')  # Why it doesn't work
    multispace_regex_2 = re.compile(r'\s-\s')  # Why it doesn't work
    df.replace({r'[^\x00-\x7F]+': ''}, regex=True, inplace=True)

    for column in df.columns:
        if column == 'instance_id':
            continue
        df[column] = df[column].str.lower().str.replace(irrelevant_regex, ' ').str.replace(multispace_regex, ' ')

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
                    'accessories', 'brand', 'builtin', 'dvd', 'batt', 'walmart.com', 'ebay', 'gaming', 'windows',
                    'laptop', 'sealed', 'wifi', 'best', 'topendelectronics', 'cd'] #, 'g']
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
                    return res

        if brand == 'asus':
            regex = [r'\sux...-.....', r'\sux.{3,5}-?.....?']  # There is a problem here TODO
            for r in regex:
                cr = re.compile(r)
                if re.search(cr, t):
                    return re.search(cr, t).group()

        if brand == 'lenovo':
            regex = [r'\sx\d{3}\s?tablet?\s?\d{0,4}', r'\sx\d{3}\s?laptop?\s?\d{0,4}', r'\sx\d{3}\s?\d{0,4}',
                     r'\sx\d{1}\scarbon\s\d{4}', r'\sx\d{1}\scarbon touch\s\d{4}']

            for r in regex:
                cr = re.compile(r)
                if re.search(cr, t):
                    res = re.search(cr, t).group()
                    for w in ['carbon', 'touch', 'tablet', 'laptop']:
                        res = res.replace(w, '')
                    return res

        if brand == 'hp':
            regex = [r'\sfolio\s?\d{4}.', r'\selitebook\s?-?\d{3,4}.',
                     r'\s\d{2}-.{4,6}', r'hp\s?\d{4}.', r'\spavilion\s?..\s?.{5}',
                     r'\s?compaq\s?.{5}', r'\s?hp\s?15\s?[a-z]\d{3}[a-z]{1,2}', r'\shp\s?12\s?.{5}',
                     r'\s?elitebook\srevolve\s?\d{3}\s?', '\s.\d{3}[pgmwm][pgmwm]?']
            for r in regex:
                cr = re.compile(r)
                if re.search(cr, t):
                    res = re.search(cr, t).group()
                    for w in ['folio', 'elitebook', 'hp', 'compaq', 'pavilion', ' 15 ', ' 12 ', 'revolve']:
                        res = res.replace(w, '')
                    return res

        if brand == 'dell':
            regex = [r'\s[nmi]\d{3,4}(-\d{4})?', r'\sinspiron\s15?\s?\d{4}', r'\sinspiron\s17?.?\s?\d{4}',
                     r'\slatitude\s15?\s?\d{4}']
            for r in regex:
                cr = re.compile(r)
                if re.search(cr, t):
                    res = re.search(cr, t).group()
                    for w in ['inspiron', '15', '17r', '   ', '  ', 'latitude']:
                        res = res.replace(w, '')
                    return res

        return None

    df['model'] = df.apply(assign_laptop_model, axis=1)

    '''
    Possible implementation for HDD and SSD:
        if ssd is found in title, hdd capacity, ssd capacity, then SSD code
        else if hdd is found in title, hdd capacity, ssd capacity, then HDD code

    '''

    def assign_ssd_capacity(record):  # picks up a bit more SSDs for some reason
        ssd = str(record['ssd_capacity']).replace(' ', '')
        ssd2 = str(record['title']).replace(' ', '')
        hdd = str(record['hdd_capacity']).replace(' ', '')
        hdd2 = str(record['title']).replace(' ', '')

        if re.search(r"(ssd)?\d{2,4}gb(ssd)?", ssd):
            return str(re.findall("\d{2,4}g", ssd)[0][:-1]) + ' gb'
        if re.search(r"(ssd)?\d{2,4}gbssd", ssd2):
            return str(re.findall("\d{2,4}gbssd", ssd2)[0][:-5]) + ' gb'
        if re.search(r"(ssd)?\dtb(ssd)?", ssd):
            return str(re.findall("\dt", ssd)[0][1]) + '000 gb'
        if re.search(r"(ssd)?\dtb(ssd)?", ssd2):
            return str(re.findall("\dtb", ssd2)[0][1]) + '000 gb'
        if re.search(r"(\d{2,4}\s?gb\s?ssd)", hdd):
            return str(re.findall("\d{2,4}\s?gb", hdd)[0][:-2]) + ' gb'
        return None

    df['ssd_capacity'] = df.apply(assign_ssd_capacity, axis=1)

    def assign_hdd_capacity(record):
        s = str(record['hdd_capacity']).replace(' ', '')
        s2 = str(record['title'].replace(' ', ''))

        if 'ssd' in s:
            return None
        if record['ssd_capacity']:
            return None

        if re.search("\d{3}gb", s):
            return str(re.findall("\d{3}gb", s)[0][:-2]) + ' gb'
        if re.search("\dtb", s):
            return str(re.findall("\dtb", s)[0][:-2] + '000') + ' gb'
        if re.search("\d{3}gb\s?(hdd)?", s2):
            return str(re.findall("\d{3}gb", s2)[0][:-2]) + ' gb'
        if re.search("hdd\dtb\s?(hdd)?", s2):
            return str(re.findall("hdd\dtb", s2)[0][3:4] + '000') + ' gb'
        return None

    df['hdd_capacity'] = df.apply(assign_hdd_capacity, axis=1)

    # ram capacity
    def assign_ram_capacity(record):
        s = str(record['ram_capacity'])
        t = str(record['title'])
        regex = re.compile(r'(\d{1,3})\s?([gm]b)')  # rare chance of encountering MB as a typo
        m = None
        # ram_c = df['ram_capacity'].str.extract(regex)
        # title_ram = df['title'].str.extract(regex)
        if s:
            m = re.search(regex, s)
        if m is None:
            m = re.search(r"\d{1,3}\s?([gm]b)\s?((ram)|(ddr\s?3)?)", t)  # r"\d{1,3}\s?([gm]b)\s?(ram)|(ddr\s?3)"

        if m is None:
            return None
        else:
            m = m.group()
            m = re.sub(r'([gm]b)', "gb", m)
            return re.sub(r'(ram)|(ddr\s?3)', "", m)

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
        regex = re.compile(r"e\d?-\d{3}")
        regex2 = re.compile(r"e\d?")
        old = re.compile(r"e-[0-9]{3}")

        for cpu in cpu_list:
            if record['cpu_type'] is not None and cpu in str(record['cpu_type']):
                return cpu

            if record['cpu_model'] is not None and cpu in str(record['cpu_model']):
                return cpu
            if record['cpu_frequency'] is not None and cpu in str(record['cpu_frequency']):
                return cpu

            if cpu in str(record['title']):
                return cpu
            if record["cpu_brand"] == 'amd':
                if re.search(regex, record['title']):
                    m= re.findall(regex, record['title'])[0]
                    return re.findall(regex2, m)[0]

                if record['cpu_model'] is not None and re.search(regex, record['cpu_model']):
                    m = re.findall(regex, record['cpu_model'])[0]
                    return re.findall(regex2, m)[0]

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
                return re.sub(r'ghz', "", m) + 'ghz'
        if re.search("ghz", record['title']):
            m = re.search(regex, record['title'])
            if m is not None:
                m = m.group()
                return re.sub(r'ghz', "", m) + 'ghz'
        if m is None:
            return None

    df['cpu_frequency'] = df.apply(assign_cpu_frequency, axis=1)
    # Unit stand. in weight
    def assign_weight(record): # TODO: Convert kg to lb if needed
        regex=re.compile('.?(\d{1,2}\.\d{1,2})\s?[lpk]')
        s = record['weight']
        m = None
        if s:
            m = re.search(regex, s)
        if m is None:
            m = re.search(regex, record['title'])
        if m is None:
            return None
        else:
            m = m.group()
            return re.sub(r"\s?[lpk]", "", m)

    #df['weight'] = df.apply(assign_weight, axis=1) # might need to fix. Not needed now

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

    def assign_new_title(record):
        # Remove extracted data from the title
        brand_tokens = record['new_title_tokens']
        str1=""
        for t in brand_tokens:
            if t == (record['brand'] or record['cpu_brand'] or record['cpu_model'] or record['cpu_frequency']
                     or record['ram_capacity'] or record['hdd_capacity'] or record['ssd_capacity']):
                t=""
            str1+=t + " "

        return str1

        # Remove model name

        #record['new_title'] = record['nwe_titl']

        # Remove brand
        # Remove screen size
        # Remove cpu brand
        # Remove cpu type
        # Ram capacity, hdd capacity, ssd capacity
    df['new_title'] = df.apply(assign_new_title, axis=1)

    return df

def clean_products_dataset(x_org):
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

