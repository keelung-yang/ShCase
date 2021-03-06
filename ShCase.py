#!/usr/bin/env python3

'''
Shanghai COVID-19 2022
Usage:
   ./ShCase.py --help
History:
    1. Initial creation, 2022-05-01
    2. Extract text by element path, 2022-05-05
    3. Report case numbers in detail, 2022-05-08
    4. Report case density, 2022-05-10
'''

import re
import time
import datetime
import pickle
import pprint
import random
import argparse
import logging

import requests
import numpy as np
import pandas as pd

from pathlib import Path
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
        return Path(path).write_text(text, encoding=encoding)
    else:
        return 0


def get_tree(url, **kwargs):
    if isinstance(url, str) and url.startswith('http'):
        encoding, text = get_html(url, **kwargs)
    else:
        text = Path(url).read_bytes()
    if text:
        return html_tree.fromstring(text)
    else:
        return None


def daily_url(start_date, end_date, delay=(0.3, 1)):
    if start_date > end_date:
        raise Exception(f'{start_date=} > {end_date=}')
    page = 0; base = 'https://wsjkw.sh.gov.cn'
    while True:
        page += 1
        url = urljoin(base, '/yqtb/index{}.html'.format(f'_{page}' if page > 1 else ''))
        time.sleep(random.uniform(*delay))
        logging.info(f'Downloading {url}')
        if (tree := get_tree(url)) is None:
            break
        for span in tree.xpath('//body/div/div/div/div/ul/li/span[@class="time"]'):
            a = span.xpath('../a')[0]
            url = a.attrib['href']
            title = a.attrib['title']
            pub_date = datetime.date.fromisoformat(span.text)
            if pub_date >= start_date and pub_date <= end_date:
                if pub_date > datetime.date(2022, 3, 18):
                    if '???????????????' in title:
                        if url.startswith('/'):
                            url = urljoin(base, url)
                        yield pub_date, url, title
                elif re.search(r'????????????\S+\d+???', title):
                    yield pub_date, urljoin(base, a.attrib['href']), title
            else:
                return


def case_numbers(text):
    if (i := text.find('????????????')) > 0:
        text = text[:i]
    date = re.findall(r'?????????(\d+)???(\d+)???(\d+)???', text)[0]
    date = datetime.date(*map(int, date))
    new_case, new_ignore, transfer, ctrl_case, ctrl_ignore = 0, 0, 0, 0, 0
    if m:= re.search(r'????????????????????????????????????(\d+)[???]?(???.*????)??????????????????????(\d+)???', text):
        new_case = int(m.group(1)); new_ignore = int(m.group(3))
    elif m:= re.search(r'????????????????????????????????????(\d+)???', text):
        new_case = int(m.group(1))
    if new_ignore ==  0 and (m := re.search(r'??????????????????????????????(\d+)???', text)):
        new_ignore = int(m.group(1))
    if m := re.search(r'(\d+)???(??????)????????????????(??????|??????)[???]?????????????????????????', text):
        transfer = int(m.group(1))
    elif m := re.search(r'????????????????????????????????????(\d+)???', text):
        transfer = int(m.group(1))
    elif m := re.search(r'(\d+)??????????????????????????????????????????', text):
        transfer = int(m.group(1))
    if m := re.search(r'(\d+)???(??????)????????????????(\d+)?????????????????????????????????????????????', text):
        ctrl_case = int(m.group(1)); ctrl_ignore = int(m.group(3))
    elif m := re.search(r'(\d+)??????????????????????????????????????????????????????????????????????????????????????????', text):
        ctrl_case = new_case; ctrl_ignore = new_ignore - int(m.group(1))
    elif m := re.search(r'(\d+)??????????????????????????????????????????????????????????????????????????????', text):
        ctrl_case = new_case; ctrl_ignore = new_ignore - int(m.group(1))
    elif m := re.search(r'(\d+)????????????????????????????????????????????????????????????', text):
        ctrl_case = new_case - int(m.group(1)); ctrl_ignore = new_ignore
    elif m := re.search(r'??????????????????????????????', text):
        ctrl_case = new_case; ctrl_ignore = new_ignore
    else:
        logging.warning(f'No insulated case found on {date}')
    return date, new_case, new_ignore, transfer, ctrl_case, ctrl_ignore


def district_detail(lines):
    dist_name = lines[0]
    new_case = 0; new_ignore = 0
    if m := re.search(r'(\d+)?????????(????????????)???????', lines[1]):
        new_case = int(m.group(1))
    elif m := re.search(r'(\d+)?????????????????????', lines[1]):
        new_case = int(m.group(1))
    elif m := re.search(r'(\d+)???????????????????????????', lines[1]):
        new_case = int(m.group(1))
    elif m := re.search(r'(\d+)???(??????)?????????????', lines[1]):
        new_case = int(m.group(1))
    elif m := re.search(r'(??????)?????????????????????????(\d+)???', lines[1]):
        new_case = int(m.group(2))
    if m := re.search(r'(\d+)[???]?(??????)??????????', lines[1]):
        new_ignore = int(m.group(1))
    elif m := re.search(r'(??????)???????????????????(\d+)???', lines[1]):
        new_ignore = int(m.group(2))
    ends = [
        r'???(??????????????????)???????(??????)???????',
        r'?????????????????????????????????????????????',
        r'?????????(??????)????????????????(??????)??????????????????'
    ]
    if [re.search(x, lines[1]) for x in ends] != [None] * len(ends):
        return dist_name, (new_case, new_ignore), []
    for i in range(2, len(lines)):
        if [re.search(x, lines[i]) for x in ends] != [None] * len(ends):
            return dist_name, (new_case, new_ignore), lines[2:i]
    return dist_name, (new_case, new_ignore), lines[2:]


def extract_text(date, root):
    dists = []
    ns = {'re': 'http://exslt.org/regular-expressions'}
    xpath = '//strong[re:match(.//text(), "(???\s*)$")]'
    if heads := [x for x in root.xpath(xpath, namespaces=ns)]:
        head = heads[0]
        nav = f'{head.getparent().tag}/{head.getparent().getparent().tag}'
        if nav == 'p/div':
            if date == datetime.date(2022, 4, 2):   # Fix duplicated content in official page
                dup_node = root.xpath('./p/span[starts-with(text(), "??????????????????")]')[-1]
                for x in dup_node.getparent().xpath('./following-sibling::*'):
                    x.getparent().remove(x)
                dup_node.getparent().remove(dup_node)
            lines = head.xpath('../../p')
            heads = [x.getparent() for x in heads]
            indices = [i for i in range(len(lines)) if lines[i] in heads]
            indices = [0] + indices + [None]
            dists = [lines[indices[i]:indices[i+1]] for i in range(len(indices)-1)]
            dists = [[x.text_content() for x in dist] for dist in dists]
        elif nav == 'span/section':
            dists = [[root.xpath('./section[1]//section/p[1]')[0].text_content()]]
            if date == datetime.date(2022, 4, 1):
                dists += [x.xpath('../../../following-sibling::*//text()') for x in heads]
            else:
                for x in heads:
                    if dist := x.xpath('../../../../../../../following-sibling::*//text()'):
                        dists.append(dist)
                    elif dist := x.xpath('../../../../../../../../following-sibling::*//text()'):
                        dists.append(dist)
                    else:
                        raise Exception(f'Failed to find data for {x.text_content()}')
        elif nav == 'section/section':
            dists = [[root.xpath('./section[1]//section/p[1]')[0].text_content()]]
            dists += [x.xpath('../../../../../../following-sibling::*//text()') for x in heads]
            if date == datetime.date(2022, 4, 25):
                dists[-1] = [''.join(dists[-1][:3])]

    # Strip lines
    for i in range(len(dists)):
        dists[i] = [re.sub(r'^[\s?????????\xa0]+|[\s?????????\xa0]+$', '', x) for x in dists[i]]
        dists[i] = [x for x in dists[i] if x]
    
    # Set dist[0] = dist_name
    for i in range(1, len(dists)):
        dist_name = heads[i-1].text_content()
        if dists[i][0] == '???????????????????????????':
            dists[i][0] = dist_name
        elif m := re.search(r'^(\d+???)?(\d+???\d+???)', dists[i][0]):
            dists[i].insert(0, dist_name)

    # Fix missing year in summary
    for dist in dists[1:]:
        if m := re.search(r'^(\d+???)?(\d+???\d+???)(????0-24???????)?', dist[1]):
            if m.group(1) is None:
                dist[1] = f'{date.year}???{dist[1]}'
            if m.group(3):
                dist[1] = re.sub(r'^(\d+???\d+???\d+???)(????0-24???????)', r'\1', dist[1])

    # Set dist[1] = dist_summary
    for i in range(1, len(dists)):
        dist_name = heads[i-1].text_content()
        if m := re.search(r'^(\d+???\d+???\d+???)', dists[i][1]):
            if m.group(0) == dists[i][1] and dists[i][2].startswith(dist_name):
                dists[i][1] = dists[i][1] + '???' + dists[i][2]
                dists[i].pop(2)
            elif dists[i][1].endswith(dist_name) or dists[i][1].endswith(dist_name[:-1]):
                dists[i][1] = dists[i][1] + dists[i][2]
                dists[i].pop(2)

    return dists


def daily_data(date, url):
    '''
    summary: (date, new_case, new_ignore, transfer, ctrl_case, ctrl_ignore, text)
    districts: (district, (new_case, new_ignore), [address])
    '''
    sh = (); districts=[]
    logging.info(f'Parsing {url}')
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
        text = root.text_content()
        date = date + datetime.timedelta(days=-1)
        new_case = 0; new_ignore = 0
        summary = ''
        if m := re.search(r'(\d+)???(\d+)???(\d+)???0???24??????????????????????????????????????????(\d+)???', text):
            date = datetime.date(*map(int, m.groups()[:3]))
            new_case = int(m.group(4))
            summary += m.group(0)
        if m := re.search(r'(\d+)???(\d+)???(\d+)???0???24??????(??????????????????????????????(\d+)???)', text):
            date = datetime.date(*map(int, m.groups()[:3]))
            new_ignore = int(m.group(5))
            summary += '???' + m.group(4) if summary else m.group(0)
        sh = (date, new_case, new_ignore, 0, 0, 0, summary)
        return sh, districts
    elif date <= datetime.date(2022, 3, 19):
        summary = root.xpath('./p[1]')[0].text_content()
        sh = case_numbers(summary) + (summary,)
        return sh, districts
    else:
        if dists := extract_text(date, root):
            summary = dists[0][0]
            sh = case_numbers(summary) + (summary,)
            districts = [district_detail(x) for x in dists[1:]]
            districts = [x for x in districts if x]
            return sh, districts
        else:
            raise Exception(f'Failed to parse ({date}) {url}')


def load_pickle(path):
    path = Path(path)
    if path.exists():
        return pickle.loads(path.read_bytes())
    else:
        return None


def save_pickle(path, obj, txt=False, encoding=None, protocol=pickle.HIGHEST_PROTOCOL):
    path = Path(path)
    path.write_bytes(pickle.dumps(obj, protocol=protocol))
    if txt:
        path.with_suffix('.txt').write_text(pprint.pformat(obj), encoding=encoding)


def update_urls(path, start_date, end_date, delay):
    if start_date > end_date:
        start_date, end_date = end_date, start_date
    urls = load_pickle(path) or {}
    dates = sorted(urls.keys())
    if dates:
        old = dates[0]; new = dates[-1]
        if start_date >= old and end_date <= new:
            return urls
        if start_date >= old and start_date <= new:
            start_date = new
        if end_date >= old and end_date <= new:
            end_date = old
    logging.info(f'Downloading urls from {start_date} to {end_date}')
    if new_urls := daily_url(start_date, end_date, delay):
        urls |= {x[0]: (x[1], x[2]) for x in new_urls}
        save_pickle(path, urls, txt=True, encoding='utf8')
    else:
        logging.warning(f'No data published from {start_date} to {end_date}')
    return urls


def update_cases(path, urls, cache_dir):
    cache_dir = Path(cache_dir)
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
            logging.info(f'{sh[0]}, {sh[1:-1]}, ({sh[1]-sh[3]-sh[4]}, {sh[2]-sh[5]})')
        else:
            logging.warning(f'No data found on {date}')
    if changed:
        save_pickle(path, cases, txt=True, encoding='utf8')
    return cases


def sort_addr(addr_list, days, sep='\t'):
    ser = pd.Series(addr_list, dtype=str).value_counts()
    grouped = ser.groupby(by=ser.values)
    columns=[f'???????????????{days}???', f'???????????????{ser.size}???', '????????????']
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
        columns=['??????', '????????????????????????', '????????????', '????????????', '????????????'],
        index=range(1, len(addr_list)+1)
    )
    df_detail = pd.DataFrame.from_dict(addr_detail, orient='index').transpose()
    df_detail.index += 1
    df_detail.columns=[k + f' ({len(addr_detail[k])})' for k in addr_detail]
    return df_summary, df_detail


def save_addr(save_to, df):
    days = df.shape[1]
    with pd.ExcelWriter(save_to) as writer:
        df1, df2 = daily_addr(df)
        df1.to_excel(writer, sheet_name='????????????')
        df.to_excel(writer, sheet_name='????????????')
        df2.to_excel(writer, sheet_name='????????????')

        addr_list = [x.strip() for x in df.values.flatten() if pd.notnull(x)]
        df3 = sort_addr(addr_list, days, '\t')
        df3.to_excel(writer, sheet_name='????????????')

        df4 = df.replace(to_replace=r'^((?!??????).)*$', value=np.nan, regex=True)
        df4 = df4.apply(lambda x: pd.Series(x.dropna().values))
        addr_list = [x.strip() for x in df4.values.flatten() if pd.notnull(x)]
        df4 = sort_addr(addr_list, days, '\n')
        df4.index += 1
        df4.to_excel(writer, sheet_name='????????????')


def save_report(save_to, cases, engine=None):
    path = Path(save_to)
    path.mkdir(exist_ok=True)
    city = []
    dists = [
        '????????????', '?????????', '?????????', '?????????', '?????????', '?????????', '?????????', '?????????', 
        '?????????', '?????????', '?????????', '?????????', '?????????', '?????????', '?????????', '?????????'
    ]
    dists = {k:[] for k in dists}
    for _, (sh, districts) in sorted(cases.items()):
        # sh: date, case, ignore, transfer, ctrl_case, ctrl_ignore, title
        total_numbers = sh[1] + sh[2]
        out_case = sh[1] - sh[3] - sh[4]
        out_ignore = sh[2] - sh[5]
        case_percent = f'{sh[1] / total_numbers :.1%}' if total_numbers else ''
        total_addr = sum(len(x[-1]) for x in districts)
        density = f'{total_numbers / total_addr :.2f}' if total_addr else ''
        city.append(list(sh[:-1]))
        city[-1] += [out_case, out_ignore, total_numbers, case_percent, total_addr, density]
        for name, (case, ignore), addr in districts:
            date = sh[0]
            total = case + ignore
            case_percent = f'{case / total :.1%}' if total else ''
            density = f'{total / len(addr) :.2f}' if addr else ''
            if addr:
                col = f'{date.month}.{date.day} ({case}+{ignore})/{len(addr)}={density}'
            else:
                col = f'{date.month}.{date.day} ({case}+{ignore})={case+ignore}'
            dists[name].append(
                (date, case, ignore, total, case_percent, len(addr), density, col, addr)
            )
    suffix = 'ods' if str(engine).lower() == 'odf' else 'xlsx'
    with pd.ExcelWriter(f'{save_to}/?????????.{suffix}', date_format='YYYY-MM-DD') as writer:
        if city:
            logging.info(f'Generating case numbers report from {city[0][0]} to {city[-1][0]}')
            df_city = pd.DataFrame.from_records(city)
            df_city.index += 1
            df_city.columns = [
                '??????', '??????', '?????????', 
                '????????????', '????????????', '???????????????', '????????????', '???????????????', 
                '??????', '????????????', '????????????', '????????????'
            ]
            df_city.to_excel(writer, sheet_name='?????????')
        for name in dists:
            logging.info(f'Generating address report of {name}')
            data = [x[:-2] for x in dists[name]]
            df_dist = pd.DataFrame.from_records(data)
            df_dist.index += 1
            df_dist.columns = ['??????', '??????', '?????????', '??????', '????????????', '????????????', '????????????']
            df_dist.to_excel(writer, sheet_name=name)
            data = dict(x[-2:] for x in dists[name])
            df_addr = pd.DataFrame.from_dict(data=data, orient='index').transpose()
            df_addr.index += 1
            save_addr(f'{save_to}/{name}.{suffix}', df_addr)


def clean(target):
    import os, shutil
    def rm(paths):
        if isinstance(paths, str) or isinstance(paths, Path):
            paths = [paths]
        for path in [Path(x) for x in paths if os.path.exists(x)]:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    if target == 'all':
        rm(['html', 'reports', 'urls.pkl', 'urls.txt', 'cases.pkl', 'cases.txt'])
    elif target == 'url':
        rm(['urls.pkl', 'urls.txt'])
    elif target == 'case':
        rm(['cases.pkl', 'cases.txt'])
    elif target == 'html':
        rm('html')
    elif target == 'report':
        rm('reports')


def setup_logger(filename, level, fmt=None, encoding='utf8'):
    datefmt='%Y-%m-%d %H:%M:%S'
    fmt = fmt if fmt else '%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s'
    fmtter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    fileHandler = logging.FileHandler(filename)
    fileHandler.setFormatter(fmtter)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(fmtter)
    handlers = [fileHandler, consoleHandler]
    logging.basicConfig(handlers=handlers, level=level)
    logging.captureWarnings(True)
    return handlers


def parse_args():
    parser = argparse.ArgumentParser(description='Shanghai COVID-19 2022')
    parser.add_argument('-s', '--startdate',
        type=datetime.date.fromisoformat,
        default=datetime.date(2022, 2, 21),
        help='Start date (YYYY-MM-DD)',
    )
    parser.add_argument('-e', '--enddate',
        type=datetime.date.fromisoformat,
        default=datetime.date.today(),
        help='End date (YYYY-MM-DD, Inclusive)',
    )
    parser.add_argument('-d', '--delay',
        type=float,
        nargs='+',
        default=[0.3, 1],
        help='A random range of delay time for downloading pages',
    )
    parser.add_argument('-ss', '--spreadsheet',
        default='odf',
        choices=['odf', 'openpyxl', 'xlsxwriter'],
        help='Spreadsheet engine for Pandas DataFrame',
    )
    parser.add_argument('-c', '--clean',
        choices=['url', 'case', 'html', 'report', 'log', 'all'],
        help='Clean local data'
    )
    parser.add_argument('-l', '--loglevel',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level'
    )
    return parser.parse_args()


def main(path, args):
    log_path = path.with_suffix('.log')
    if args.clean:
        clean(args.clean)
        if args.clean in ['log', 'all']:
            log_path.unlink(missing_ok=True)
            print(args)
        return
    setup_logger(log_path, args.loglevel)
    logging.info(str(args))

    if len(args.delay) == 1:
        delay = (0, args.delay[0])
    else:
        delay = tuple(args.delay[:2])
    urls = update_urls('urls.pkl', args.startdate, args.enddate, delay)
    cases = update_cases('cases.pkl', urls, 'html')
    save_report('reports', cases, args.spreadsheet)


if __name__ == '__main__':
    import sys
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        fullpath = Path(sys.executable)
    else:
        fullpath = Path(__file__).resolve()

    try:
        args = parse_args()
        main(fullpath, args)
    except Exception as ex:
        logging.exception(str(ex))
    finally:
        logging.info('Done\n')
