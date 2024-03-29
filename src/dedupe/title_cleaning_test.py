
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
    sp = spacy.load('en_core_web_sm')

    # Read helper datasets stats
    extra_brands = {'acer', 'google', 'toshiba', 'dell', 'xiaomi', 'asus', 'mediacom', 'hp', 'vero', 'lg',
                    'chuwi', 'lenovo', 'apple', 'microsoft', 'fujitsu', 'huawei', 'samsung', 'razer', 'msi'}
    screen_sizes = {'14.1', '15.6', '12.5', '12', '15', '13.5', '14', '17', '17.3',
                    '12.3', '13.9', '15.4', '10.1', '13', '11.6', '18.4', '13.3', '11.3'}

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

    remove_words = {'"', '&', '(', ')', '.com', '/', ':', 'accessories', 'aluminum', 'amazon.com', 'america',
                    'and', 'at', 'audio', 'audiophile', 'backlight', 'batt', 'beats', 'bes', 'best', 'bluetooth',
                    'bluray', 'brand', 'built-in', 'builtin', 'burner', 'buy.net', 'canada', 'card', 'cd',
                    'certified', 'clarinet', 'comfyview', 'comparison', 'computer', 'computers', 'cool', 'core',
                    'deals', 'dimm', 'display', 'downgrade', 'drive', 'dualcore', 'dvd', 'dvdrw', 'dvdwriter',
                    'ebay', 'edition', 'end', 'finger', 'for', 'french', 'gaming', 'graphics', 'high', 'home',
                    'internationalaccessories', 'keyboard', 'lan', 'laptop', 'micro', 'microphone', 'mini.ca',
                    'new', 'notebook', 'nx.m8eaa.007', 'overstock.com', 'pc', 'performance', 'portable', 'premium',
                    'price', 'professional', 'refurbished', 'revolve', 'sale', 'screen', 'sd', 'sealed', 'slot',
                    'speaker', 'special', 'supermulti', 'switching', 'technology', 'thenerds.net', 'tigerdirect',
                    'topendelectronics', 'touch', 'ultrabase', 'ultraportable', 'us', 'voice', 'vology', 'walmart',
                    'walmart.com', 'webcam', 'wifi', 'win', 'windows', 'wireless', 'with'}  #, 'g']
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

    def assign_ssd_capacity(record):
        ssd = str(record['ssd_capacity']).replace(' ', '')
        ssd2 = str(record['title']).replace(' ', '')
        hdd = str(record['hdd_capacity']).replace(' ', '')

        if re.search("\d{3,4}gb", ssd):
            return int(re.findall("\d{3,4}gb", ssd)[0][:-2])
        if re.search("\dtb", ssd):
            return int(re.findall("\dtb", ssd)[0][:-2] + '000')
        if re.search("\d{3,4}gbssd", ssd2):
            return int(re.findall("\d{3,4}gbssd", ssd2)[0][:-5])
        if re.search("ssd\d{3,4}gb", ssd2):
            return int(re.findall("ssd\d{3,4}gb", ssd2)[0][3:-2])
        if re.search("ssd\dtb", ssd2):
            return int(re.findall("ssd\dtb", ssd2)[0][3:4] + '000')
        if re.search("\dtbssd", ssd2):
            return int(re.findall("\d{1}tbssd", ssd2)[0][0] + '000')
        if re.search(r"\d{2,4}\s?gb\s?ssd", hdd):
            return str(re.findall("\d{2,4}\s?gb", hdd)[0][:-2])
        return None

    df['ssd_capacity'] = df.apply(assign_ssd_capacity, axis=1)

    def assign_hdd_capacity(record):
        s = str(record['hdd_capacity']).replace(' ', '')
        s2 = str(record['title'].replace(' ', ''))

        if 'ssd' in s:
            return None
        if record['ssd_capacity']:
            return None

        if re.search("\d{3}gb", s): #Possible bug with spaces but it works for now without \s?
            return str(re.findall("\d{3}gb", s)[0][:-2])
        if re.search("\dtb", s):
            return str(re.findall("\dtb", s)[0][:-2] + '000')
        if re.search("\d{3}gb\s?(hdd)?", s2):
            return str(re.findall("\d{3}gb", s2)[0][:-2])
        if re.search("hdd\dtb\s?(hdd)?", s2):
            return str(re.findall("hdd\dtb", s2)[0][3:4] + '000')
        return None

    df['hdd_capacity'] = df.apply(assign_hdd_capacity, axis=1)
    # ram capacity
    def assign_ram_capacity(record):
        cap = str(record['ram_capacity'])
        title = str(record['title'])
        rType = str(record['ram_type'])
        regex = re.compile(r'(\d{1,3})\s?([gm]b)')  # rare chance of encountering MB as an error
        # regex2 = re.compile(r"\d{1,3}\s?(gb)\s?((ram)|(ddr\s?3)?)")
        m = None
        # ram_c = df['ram_capacity'].str.extract(regex)
        # title_ram = df['title'].str.extract(regex)
        if cap:
            m = re.search(regex, cap)
        if m is None:
            m = re.search(regex, rType)
        if m is None:
            m = re.search(regex, title)  # r"\d{1,3}\s?([gm]b)\s?(ram)|(ddr\s?3)"
        if m is None:
            return None
        else:
            m = m.group()
            m = re.sub(r'\s?([gm]b)', "", m)
            # m = re.sub(r'(ram)|(ddr\s?3)', "", m)
            if m in [record['hdd_capacity'], record['ssd_capacity']]:  # Picks up HDD for RAM. Fix doesn't work
                # print("Broke")
                return None
            return m

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

    def assign_model_number(record):
        '''
        TODO
        if "x230" in org_title and "3435" in org_title:
                        mod_item["model"] = "3435"

                    if "hp" in org_title:
                        #regex for specific HP laptops
                        hp_li = hp_new_model.findall(org_title)
                        if len(hp_li) > 0:
                            mod_item["model"] = " ".join(hp_li[0].replace("-","").replace(" ","").split())

                    if "hp" in org_title and "revolve" in org_title and "810" in org_title:
                        mod_item["model"] = "revolve 810 "
                        if "g1" in org_title.lower():
                            mod_item["model"] += "g1"
                        elif "g2" in org_title.lower():
                            mod_item["model"] += "g2"

                    if "hp" in org_title and "compaq" in org_title and "nc6400" in org_title:
                        mod_item["model"] = "nc6400"

                    if "lenovo" in org_title or "thinkpad" in org_title:
                        tp_li = lenovo_thinkpad_model.findall(org_title)
                        if len(tp_li) > 0:
                            mod_item["model"] = " ".join(tp_li[0].split())
        '''
        return "232";

    # df['model_number'] = df.apply(assign_model_number)

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
                    m = re.findall(regex, record['title'])[0]
                    return re.findall(regex2, m)[0]

                if record['cpu_model'] is not None and re.search(regex, record['cpu_model']):
                    m = re.findall(regex, record['cpu_model'])[0]
                    return re.findall(regex2, m)[0]
        return None

    df['cpu_type'] = df.apply(assign_cpu_type, axis=1)

    def assign_cpu_model(record):
        model = record['cpu_model']
        regex = re.compile(r"-?\d{1,4}([mu])")  # For intel cpus
        regex2 = re.compile(r"[ea]\d?-\d{1,4}[m]?")  # for amd A and E series. Needs detection after AMD tag in title
        m = None
        if record['cpu_brand'] == 'intel' and model is not None:
            m = re.search(regex, model)
            if m is not None:
                m = m.group()
                return re.sub(r'-', "", m)
            m = re.search(r"m\d{3,4}", model)  # TODO Used to pick up i7-M640. Double check
            if m is not None:
                return m.group()
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

    # def assign_cpu(record):
    #     cpu_type = record['cpu_type']
    #     cpu_model = record['cpu_model']
    #     res = ""
    #     if cpu_type is not None:
    #         res += cpu_type
    #
    #     if cpu_model is not None:
    #         res += '-'
    #         res += cpu_model
    #     return res
    #
    # df['cpu_model'] = df.apply(assign_cpu, axis=1)

    def assign_new_title(record):
        # Remove extracted data from the title
        brand_tokens = record['new_title_tokens']
        str1 = ""
        for t in brand_tokens:
            if t == (record['brand'] or record['cpu_brand'] or record['cpu_model'] or record['cpu_frequency']
                     or record['ram_capacity'] or record['hdd_capacity'] or record['ssd_capacity']):
                t = ""
            str1 += t + " "

        return str1

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

