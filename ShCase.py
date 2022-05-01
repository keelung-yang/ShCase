#!/usr/bin/env python3

'''
Shanghai COVID-19 2022
Usage:
   ./ShCase.py --help
History:
    1. Initial creation. 2022-05-01
'''

import re
import datetime
import pickle
import pathlib
import pprint
import argparse
import logging

import requests
import numpy as np
import pandas as pd

from urllib.parse import urljoin
from lxml import html as html_tree


def get_html(url, **kwargs):
    try:
        if 'User-Agent' not in kwargs:
            headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64)'}
        res = requests.get(url, headers=headers, **kwargs)
        res.raise_for_status()
        return res.encoding, res.text
    except requests.exceptions.HTTPError as ex:
        logging.error(f'Failed to download! {ex}')
        return None, None
    except:
        logging.error(f'Failed to download {url}')
        return None, None


def save_page(url, path, **kwargs):
    encoding, text = get_html(url, **kwargs)
    if text:
        return pathlib.Path(path).write_text(text, encoding=encoding)
    else:
        return 0


def get_tree(url, **kwargs):
    if isinstance(url, str) and url.startswith('http'):
        encoding, text = get_html(url, **kwargs)
    else:
        text = pathlib.Path(url).read_bytes()
    if text:
        return html_tree.fromstring(text)
    else:
        return None


def daily_url(start_date, end_date):
    if start_date > end_date:
        start_date, end_date = end_date, start_date
    page = 0; base = 'https://wsjkw.sh.gov.cn'
    while True:
        page += 1
        url = urljoin(base, '/xwfb/index{}.html'.format(f'_{page}' if page > 1 else ''))
        if (tree := get_tree(url)) is None:
            break
        for span in tree.xpath('//body/div/div/div/div/ul/li/span[@class="time"]'):
            a = span.xpath('../a')[0]
            url = a.attrib['href']
            title = a.attrib['title']
            pub_date = datetime.date.fromisoformat(span.text)
            if pub_date >= start_date and pub_date <= end_date:
                if pub_date > datetime.date(2022, 3, 18):
                    if '居住地信息' in title:
                        if url.startswith('/'):
                            url = urljoin(base, url)
                        yield pub_date, url, title
                elif re.search(r'新增本土\S+\d+例', title):
                    yield pub_date, urljoin(base, a.attrib['href']), title
            else:
                return


def sh_summary(text):
    lines = [x for x in text.split('，') if not ('境外' in x or re.fullmatch(r'\D+', x))]
    date = re.findall(r'通报：(\d+)年(\d+)月(\d+)日', lines[0])[0]
    date = datetime.date(*map(int, date))
    new_case = int(re.findall(r'^新增本土新冠肺炎确诊病例(\d+)', lines[1])[0])
    new_ignore = int(re.findall(r'无症状感染者(\d+)例$', lines[1])[0])
    others = ' '.join(lines[2:])
    transfer = 0; ctrl_case = 0; ctrl_ignore = 0
    if m := re.search(r'(\d+)例确诊病例为(此前|既往)无症状感染者转归', others):
        transfer = int(m.group(1))
    elif m := re.search(r'无症状感染者转为确诊病例(\d+)例', lines[1]):
        transfer = int(m.group(1))
    elif m := re.search(r'(\d+)例由无症状感染者转为确诊病例', lines[1]):
        transfer = int(m.group(1))
    if m := re.search(r'(\d+)例(本土)*确诊病例\S+在隔离管控中', others):
        ctrl_case = int(m.group(1))
    if m := re.search(r'(\d+)例无症状感染者\S*在隔离管控中', others):
        ctrl_ignore = int(m.group(1))
    return date, new_case, new_ignore, transfer, ctrl_case, ctrl_ignore, text


def district_detail(lines):
    if indices := [i for i in range(len(lines)) if re.search(r'已对相关居住地落实(终末)*消毒', lines[i])]:
        lines = lines[:indices[0]]
    if lines and (m := re.search(r'\d+年\d+月\d+日，(([^区]+)区)', lines[0])):
        name = m.groups()[0]
        new_case = 0; new_ignore = 0
        if m := re.search(r'(\d+)例本土(新冠肺炎)*确诊', lines[0]):
            new_case = int(m.group(1))
        if m := re.search(r'(\d+)例本土无症状', lines[0]):
            new_ignore = int(m.group(1))
        return name, (new_case, new_ignore), lines[1:]


def daily_data(date, url):
    '''
    summary: (date, new_case, new_ignore, transfer, ctrl_case, ctrl_ignore, text)
    districts: (district, (new_case, new_ignore), [address])
    '''
    sh = (); districts=[]
    if (tree := get_tree(url)) is None:
        return sh, districts
    # https://mp.weixin.qq.com/s/xxx or
    # https://wsjkw.sh.gov.cn
    nodes = tree.xpath('//div[@id="js_content"]') or \
            tree.xpath('//div[@class="Article_content"]')   
    if nodes:
        root = nodes[0]
    else:
        return sh, districts
    if date <= datetime.date(2022, 3, 6):
        text = nodes[0].text_content()
        date = date + datetime.timedelta(days=-1)
        new_case = 0; new_ignore = 0
        summary = ''
        if m := re.search(r'(\d+)年(\d+)月(\d+)日0—24时，新增本土新冠肺炎确诊病例(\d+)例', text):
            date = datetime.date(*map(int, m.groups()[:3]))
            new_case = int(m.group(4))
            summary += m.group(0)
        if m := re.search(r'(\d+)年(\d+)月(\d+)日0—24时，(新增本土无症状感染者(\d+)例)', text):
            date = datetime.date(*map(int, m.groups()[:3]))
            new_ignore = int(m.group(5))
            summary += '；' + m.group(4) if summary else m.group(0)
        sh = (date, new_case, new_ignore, 0, 0, 0, summary)
        return sh, districts
    elif date <= datetime.date(2022, 3, 19):
        sh = sh_summary(nodes[0].xpath('./p[1]')[0].text_content())
        return sh, districts
    else:
        root = nodes[0]
        if 'id' in root.attrib and root.attrib['id'] == 'js_content':   # WeChat
            nodes = root.xpath('//section/section/section/p')
        elif date == datetime.date(2022, 4, 1):
            nodes = root.xpath('//section/section/section/p')
        else:
            nodes = root.xpath('./p')
        lines = [x.text_content() for x in nodes]
        lines = [re.sub(r'^[\s，。、\xa0]+|[\s，。、\xa0]+$', '', x) for x in lines]
        lines = [x for x in lines if x]
        if date == datetime.date(2022, 3, 29):
            for i in range(len(lines)):
                if lines[i].startswith('2022年3月28日，奉贤无新增本土确诊病例'):
                    lines[i] = lines[i].replace('奉贤', '奉贤区')
        elif date == datetime.date(2022, 4, 4):
            for i in range(len(lines)):
                if lines[i].startswith('2022年4月3日，青浦新增3例本土确诊病例'):
                    lines[i] = lines[i].replace('青浦', '青浦区')
        indices = [i for i in range(len(lines)) if re.search(r'^\d+年\d+月\d+日，[^区]+区', lines[i])]
        indices = [0] + indices + [None]
        lines = [lines[indices[i]:indices[i+1]] for i in range(len(indices)-1)]
        if summary := [x for x in lines[0] if re.search(r'通报：\d+年\d+月\d+日', x)]:
            summary = sh_summary(summary[0])
        districts = [district_detail(x) for x in lines[1:]]
        districts = [x for x in districts if x is not None]
    return summary, districts


def load_pickle(path):
    path = pathlib.Path(path)
    if path.exists():
        return pickle.loads(path.read_bytes())
    else:
        return None


def save_pickle(path, obj, txt=True, encoding=None, protocol=pickle.HIGHEST_PROTOCOL):
    path = pathlib.Path(path)
    path.write_bytes(pickle.dumps(obj, protocol=protocol))
    if txt:
        path.with_suffix('.txt').write_text(pprint.pformat(obj), encoding=encoding)


def update_urls(path, start_date, end_date):
    urls = load_pickle(path) or {}
    dates = sorted(urls.keys())
    if dates and (start_date >= dates[0] and end_date <= dates[-1]):
        return urls
    if new_urls := daily_url(start_date, end_date):
        urls |= {x[0]: (x[1], x[2]) for x in new_urls}
        save_pickle(path, urls)
    else:
        logging.warning(f'No data published from {start_date} to {end_date}')
    return urls


def update_cases(path, urls, cache_dir):
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    cases = load_pickle(path) or {}
    changed = False
    for date in urls:
        page = cache_dir / f'{date}.html'
        if not page.exists():
            save_page(urls[date][0], page)
        if date in cases:
            continue
        sh, districts = daily_data(date, page)
        if sh and (date not in cases):
            changed = True
            cases[date] = (sh, districts)
            # date, case, ignore, transfer, ctrl_case, ctrl_ignore, out_case, out_ignore
            logging.info(f'{str(sh[0]), sh[1:-1]}, {sh[1]-sh[3]-sh[4]}, {sh[2]-sh[5]}')
        else:
            logging.warning(f'No data found on {date}')
    if changed:
        save_pickle(path, cases)
    return cases


def sort_addr(addr_list, days, sep='\t'):
    ser = pd.Series(addr_list, dtype=str).value_counts()
    grouped = ser.groupby(by=ser.values)
    columns=[f'新增天数（{days}）', f'地址个数（{ser.size}）', '地址列表']
    return pd.DataFrame.from_dict(
        {k: [k, g.count(), sep.join(sorted(g.index))] for k, g in grouped},
        orient='index',
        columns=columns
    ).set_index(columns[0]).sort_index(ascending=False)


def daily_addr(df):
    days = df.shape[1]
    all_addr = set()
    addr_list = []
    addr_detail = {}
    for i in range(days):
        date = re.search(r'(\d+\.\d+)\s*', df.columns[i]).group(1)
        day_addr = set([x[0].strip() for x in df.iloc[:, [i]].values if pd.notnull(x)])
        addr_detail[date] = sorted(day_addr - all_addr)
        all_addr |= day_addr
        addr_list.append((
            date,
            len(all_addr),
            len(day_addr),
            len(addr_detail[date]),
            len(day_addr) - len(addr_detail[date])
        ))
    df_summary = pd.DataFrame(
        addr_list,
        columns=['日期', '累积地址（去重）', '日增地址', '净增地址', '重发地址'],
        index=range(1, len(addr_list)+1)
    )
    df_detail = pd.DataFrame.from_dict(addr_detail, orient='index').transpose()
    df_detail.index += 1
    df_detail.columns=[k + f' ({len(addr_detail[k])})' for k in addr_detail]
    return df_summary, df_detail


def save_addr(save_to, df, engine='odf'):
    days = df.shape[1]
    with pd.ExcelWriter(save_to, engine=engine) as writer:
        df1, df2 = daily_addr(df)
        df1.to_excel(writer, sheet_name='日增统计')
        df2.to_excel(writer, sheet_name='净增明细')

        addr_list = [x.strip() for x in df.values.flatten() if pd.notnull(x)]
        df3 = sort_addr(addr_list, days, '\t')
        df3.to_excel(writer, sheet_name='地址排序')

        df4 = df.replace(to_replace=r'^((?!工地).)*$', value=np.nan, regex=True)
        df4 = df4.apply(lambda x: pd.Series(x.dropna().values))
        addr_list = [x.strip() for x in df4.values.flatten() if pd.notnull(x)]
        df4 = sort_addr(addr_list, days, '\n')
        df4.index += 1
        df4.to_excel(writer, sheet_name='工地排序')


def save_report(save_to, cases):
    path = pathlib.Path(save_to)
    path.mkdir(exist_ok=True)
    dists = [
        '黄浦区', '徐汇区', '长宁区', '静安区', '普陀区', '虹口区', '杨浦区','宝山区', 
        '浦东新区', '闵行区', '嘉定区', '金山区', '松江区', '青浦区', '奉贤区', '崇明区'
    ]
    dists = {k:[] for k in dists}
    for _, (sh, districts) in cases.items():
        for name, (case, ignore), addr in districts:
            date = sh[0]
            col = f'{date.month}.{date.day} ({len(addr)})'
            dists[name].append((col, addr))   # name: (date, addr_list)
    for name in dists:
        logging.info(f'Generating address report of {name}')
        data = dict(dists[name])
        df = pd.DataFrame.from_dict(data=data, orient='index').transpose()
        df.index += 1
        save_addr(f'{save_to}/{name}.ods', df)


def clean(target):
    import shutil
    def rm_files(paths):
        for x in paths:
            pathlib.Path(x).unlink(missing_ok=True)
    if target == 'all':
        shutil.rmtree('html')
        shutil.rmtree('addr')
        rm_files(['urls.pkl', 'urls.txt', 'cases.pkl', 'cases.txt'])
    elif target == 'url':
        rm_files(['urls.pkl', 'urls.txt'])
    elif target == 'case':
        rm_files(['cases.pkl', 'cases.txt'])
    elif target == 'html':
        shutil.rmtree('html')
    elif target == 'addr':
        shutil.rmtree('addr')


def setup_logger(level, filename, fmt=None, encoding='utf8'):
    datefmt='%Y-%m-%d %H:%M:%S'
    fmt = fmt if fmt else '%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s'
    fmtter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    fileHandler = logging.FileHandler(filename)
    fileHandler.setFormatter(fmtter)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(fmtter)
    logging.basicConfig(handlers=[fileHandler, consoleHandler], level=level)
    logging.captureWarnings(True)


def setup_args():
    parser = argparse.ArgumentParser(description='Shanghai COVID-19 2022')
    parser.add_argument('-s', "--startdate",
        type=datetime.date.fromisoformat,
        default=datetime.date(2022, 2, 21),
        help="The Start Date - format YYYY-MM-DD",
    )
    parser.add_argument('-e', "--enddate",
        type=datetime.date.fromisoformat,
        default=datetime.date.today(),
        help="The End Date format YYYY-MM-DD (Inclusive)",
    )
    parser.add_argument('-c', '--clean',
        choices=['url', 'case', 'html', 'addr', 'log', 'all'],
        help='Clean local data'
    )
    parser.add_argument('-l', '--loglevel',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level'
    )
    return parser.parse_args()


def main(path, args):
    if args.clean:
        clean(args.clean)
        if args.clean in ['log', 'all']:
            logging.shutdown()
            path.with_suffix('.log').unlink(missing_ok=True)
        return
    urls = update_urls('urls.pkl', args.startdate, args.enddate)
    cases = update_cases('cases.pkl', urls, 'html')
    save_report('addr', cases)


if __name__ == '__main__':
    import sys
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        fullpath = pathlib.Path(sys.executable)
    else:
        fullpath = pathlib.Path(__file__).resolve()
    
    args = setup_args()
    setup_logger(args.loglevel, fullpath.with_suffix('.log'))
    logging.info(str(args))

    try:
        main(fullpath, args)
    except Exception as ex:
        logging.error(str(ex))
