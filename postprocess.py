import argparse
import json
import math
import re
import sys
from collections import Counter
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

FIELDS = [
    "Last_name_ru", "First_name_ru", "Middle_name_ru", "Sex_ru", "Birth_date",
    "Birth_place_ru", "Issue_date", "Issue_organisation_code", "Licence_number",
    "Issue_organization_ru",
]
STRUCT = [
    "Birth_date", "Sex_ru", "Issue_date", "Issue_organisation_code", "Licence_number"
]
NAMES = {
    "Last_name_ru": "last",
    "First_name_ru": "first",
    "Middle_name_ru": "middle",
}
FORMAT = {
    "Birth_date": re.compile(r"\d{2}\.\d{2}\.\d{4}"),
    "Issue_date": re.compile(r"\d{2}\.\d{2}\.\d{4}"),
    "Sex_ru": re.compile(r"(МУЖ|ЖЕН)\."),
    "Issue_organisation_code": re.compile(r"\d{3}-\d{3}"),
    "Licence_number": re.compile(r"\d{2} \d{2} \d{6}"),
}
STATUSES = [
    "agree", "conflict", "mrz_only", "agree_soft", "viz_chk", "viz_only",
    "soft_only", "none",
]
SOURCES = ['web', 'mobile', 'mobile_v2']
PREP = {'В', 'ПО', 'И', 'ОТДЕЛА', 'ПРИ'}
REF_DATE = date(2026, 10, 1)
TRANSLIT = dict(zip(
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ",
    ["A", "B", "V", "G", "D", "E", "2", "J", "Z", "I", "Q", "K", "L",
     "M", "N", "O", "P", "R", "S", "T", "U", "F", "H", "C", "3", "4",
     "W", "X", "Y", "9", "6", "7", "8"],
))
INV = {v: k for k, v in TRANSLIT.items()}

def norm(s):
    """
        Нормализуем текст для сравнения полей
    """
    return ' '.join(str(s or '').split()).upper()

def load_jsonl(path):
    """
        Загружаем JSONL и индексируем документы по ID
    """
    out = {}
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                d = json.loads(line)
                out[d['doc_id']] = d
    return out

def mn(x):
    """
        Берём минимальную уверенность символов
    """
    x = list(x)
    return min(x) if x else 0.0

def parse_date(s):
    """
        Преобразуем дату в объект date и отбрасываем некорректные значения
    """
    m = re.fullmatch('(\\d{2})\\.(\\d{2})\\.(\\d{4})', s or '')
    if not m:
        return None
    try:
        return date(int(m[3]), int(m[2]), int(m[1]))
    except ValueError:
        return None

def add_years(d, y):
    """
        Прибавляем годы и обрабатываем 29 февраля
    """
    try:
        return d.replace(year=d.year + y)
    except ValueError:
        return d.replace(year=d.year + y, day=28)

def passport_valid(birth, issue, ref=REF_DATE):
    """
        Проверяем, действует ли паспорт на контрольную дату
    """
    if birth is None or issue is None:
        return None
    if issue > ref or issue < birth:
        return False
    for age in (20, 45):
        if issue < add_years(birth, age) <= ref:
            return False
    return True

def translit(s):
    """
        Переводим русские буквы в символы MRZ
    """
    return ''.join((TRANSLIT.get(ch, '<' if ch in ' -' else '?') for ch in norm(s)))
_W = [7, 3, 1]

def _mrz_val(c):
    """
        Преобразуем символ MRZ в числовое значение
    """
    if c.isdigit():
        return int(c)
    if 'A' <= c <= 'Z':
        return ord(c) - 55
    return 0 if c == '<' else None

def mrz_check(s, digit):
    """
        Проверяем контрольную цифру по алгоритму ICAO
    """
    if not digit.isdigit():
        return False
    total = 0
    for i, c in enumerate(s):
        value = _mrz_val(c)
        if value is None:
            return False
        total += value * _W[i % 3]
    return total % 10 == int(digit)

def _ymd(s, kind):
    """
        Преобразуем дату MRZ в привычный формат
    """
    if not s.isdigit() or len(s) != 6:
        return None
    yy, mm, dd = (int(s[:2]), int(s[2:4]), int(s[4:]))
    if kind == 'birth':
        year = 1900 + yy if yy > 26 else 2000 + yy
    else:
        year = 1900 + yy if yy >= 90 else 2000 + yy
    try:
        date(year, mm, dd)
    except ValueError:
        return None
    return f'{dd:02d}.{mm:02d}.{year}'

def parse_mrz(mrz):
    """
        Разбираем MRZ и используем контрольные цифры для проверки полей
    """
    if not mrz or not mrz.get('lines') or len(mrz['lines']) < 2:
        return None
    l1, l2 = [x['text'].replace('\\<', '<') for x in mrz['lines'][:2]]
    c1, c2 = [x['char_conf'] for x in mrz['lines'][:2]]
    o = dict(l1=l1, l2=l2, hard={}, soft={}, mins={}, names=None, names_conf={}, names_confl={}, ok_len=len(l2) == 44 and len(c2) == 44, ok_num=False, ok_birth=False, ok_tail=False, trunc=len(l1) >= 44, l1min=mn(c1))
    if o['ok_len']:
        o['ok_num'] = l2[:9].isdigit() and mrz_check(l2[:9], l2[9])
        o['ok_birth'] = mrz_check(l2[13:19], l2[19])
        o['ok_tail'] = mrz_check(l2[28:41], l2[42])
        birth = _ymd(l2[13:19], 'birth')
        if birth:
            o['soft']['Birth_date'] = birth
            o['mins']['Birth_date'] = mn(c2[13:20])
            if o['ok_birth']:
                o['hard']['Birth_date'] = birth
        if l2[20] in ('M', 'F'):
            sex = 'МУЖ.' if l2[20] == 'M' else 'ЖЕН.'
            o['soft']['Sex_ru'] = o['hard']['Sex_ru'] = sex
            o['mins']['Sex_ru'] = mn(c2[20:21])
        issue = _ymd(l2[29:35], 'issue')
        if issue:
            o['soft']['Issue_date'] = issue
            o['mins']['Issue_date'] = mn(c2[28:43])
            if o['ok_tail']:
                o['hard']['Issue_date'] = issue
        if l2[35:41].isdigit():
            code = l2[35:38] + '-' + l2[38:41]
            o['soft']['Issue_organisation_code'] = code
            o['mins']['Issue_organisation_code'] = mn(c2[28:43])
            if o['ok_tail']:
                o['hard']['Issue_organisation_code'] = code
        if l2[:9].isdigit() and l2[28].isdigit():
            licence = f'{l2[0:2]} {l2[2]}{l2[28]} {l2[3:9]}'
            o['soft']['Licence_number'] = licence
            o['mins']['Licence_number'] = mn(c2[0:10] + c2[28:29] + c2[42:43])
            if o['ok_num'] and o['ok_tail']:
                o['hard']['Licence_number'] = licence
    if l1.startswith('PNRUS') and len(c1) == len(l1):
        s = l1[5:]
        i = s.find('<<')
        if i > 0:
            last = s[:i]
            first, _, middle = s[i + 2:].rstrip('<').partition('<')
            a_first = 5 + i + 2
            a_mid = a_first + len(first) + 1
            spans = {'last': (5, 5 + len(last)), 'first': (a_first, a_first + len(first)), 'middle': (a_mid, a_mid + len(middle))}
            o['names'] = {'last': last, 'first': first, 'middle': middle}
            for key, (a, b) in spans.items():
                o['names_confl'][key] = c1[a:b]
                if b > a:
                    o['names_conf'][key] = mn(c1[a:b])
                else:
                    start = a_first + len(first)
                    o['names_conf'][key] = mn(c1[start:start + 2]) or 0.0
    return o

def viz_checks(raw, mrz):
    """
        Проверяем поля VIZ с помощью контрольных цифр MRZ
    """
    result = {f: False for f in STRUCT}
    if not mrz or not mrz['ok_len']:
        return result
    l2 = mrz['l2']
    birth = raw.get('Birth_date')
    if birth and FORMAT['Birth_date'].fullmatch(birth):
        result['Birth_date'] = mrz_check(birth[8:10] + birth[3:5] + birth[0:2], l2[19])
    licence = raw.get('Licence_number')
    issue = raw.get('Issue_date')
    code = raw.get('Issue_organisation_code')
    if licence and issue and code and FORMAT['Licence_number'].fullmatch(licence) and FORMAT['Issue_date'].fullmatch(issue) and FORMAT['Issue_organisation_code'].fullmatch(code):
        d = licence.replace(' ', '')
        num9 = d[0:3] + d[4:10]
        tail = d[3] + issue[8:10] + issue[3:5] + issue[0:2] + code[0:3] + code[4:7]
        tail_ok = mrz_check(tail, l2[42])
        number_ok = mrz_check(num9, l2[9])
        result['Issue_date'] = result['Issue_organisation_code'] = tail_ok
        result['Licence_number'] = tail_ok and number_ok
    return result

def name_agree(viz_txt, mrz_part, trunc):
    """
        Сравниваем имя VIZ с транслитерацией из MRZ
    """
    a, b = (translit(viz_txt), mrz_part)
    return a == b or (trunc and (a.startswith(b) or b.startswith(a)))

def repair_name(v, vconf, m, mconf):
    """
        Восстанавливаем имя только при совпадении длины и уверенности символов
    """
    if len(v) != len(m) or len(vconf) != len(v) or len(mconf) != len(m):
        return None
    out = []
    for ch, cv, mc, cm in zip(v, vconf, m, mconf):
        if TRANSLIT.get(ch, '<' if ch in ' -' else '?') == mc:
            out.append(ch)
        elif mc in INV and cm > cv:
            out.append(INV[mc])
        else:
            out.append(ch)
    result = ''.join(out)
    return result if translit(result) == m else None

class Resources:
    """
        Словари из обучающей разметки и справочника подразделений
    """

    def __init__(self, labels, ref_path):
        """
            Подготавливаем словари из разметки и справочника подразделений
        """
        self.org = {}
        self.ref = {}
        self.tok = {k: Counter() for k in ['bp_first', 'bp_last', 'bp_word', 'org_first', 'org_last', 'org_word', 'fn_word', 'mn_word']}
        for label in labels.values():
            fields = label['fields']
            self.org.setdefault(fields['Issue_organisation_code'], Counter())[norm(fields['Issue_organization_ru'])] += 1
            for key, values in own_sets(fields).items():
                for token in values:
                    self.tok[key][token] += 1
        try:
            ref = pd.read_csv(ref_path, dtype=str).dropna()
            for code, name in zip(ref.iloc[:, 0].str.strip(), ref.iloc[:, 1]):
                self.ref.setdefault(code, set()).add(norm(name))
        except Exception as error:
            print('справочник не прочитан:', error, file=sys.stderr)

    def org_cands(self, code, own_org=None):
        """
            Собираем возможные названия органа выдачи по коду
        """
        out = {}
        counts = self.org.get(code)
        if counts:
            counts = Counter(counts)
            if own_org is not None:
                counts[norm(own_org)] -= 1
            for key, value in counts.items():
                if value > 0:
                    out[key] = True
        for key in self.ref.get(code, ()):
            out.setdefault(key, False)
        return out

    def freq(self, key, token, own_sets):
        """Считаем частоту токена без вклада текущего документа"""
        return self.tok[key][token] - (1 if own_sets and token in own_sets[key] else 0)

def own_sets(fields):
    """
        Выделяем слова из полей текущего документа
    """
    bp = norm(fields['Birth_place_ru']).split()
    org = norm(fields['Issue_organization_ru']).split()
    return {'bp_first': {bp[0]} if bp else set(), 'bp_last': {bp[-1]} if bp else set(), 'bp_word': set(bp), 'org_first': {org[0]} if org else set(), 'org_last': {org[-1]} if org else set(), 'org_word': set(org), 'fn_word': {norm(fields['First_name_ru'])}, 'mn_word': {norm(fields['Middle_name_ru'])}}

def snap_org(pred, cands):
    """
        Оцениваем сходство названия органа выдачи, не исправляя его
    """
    info = dict(org_ratio=np.nan, org_margin=np.nan, org_ncand=len(cands), org_exact=0, org_train=0)
    if not cands or pred is None:
        return (pred, info)
    if pred in cands:
        info.update(org_exact=1, org_train=int(cands[pred]), org_ratio=1.0)
        return (pred, info)
    scores = sorted((SequenceMatcher(None, pred, candidate).ratio() for candidate in cands), reverse=True)
    info.update(org_ratio=scores[0], org_margin=scores[0] - (scores[1] if len(scores) > 1 else 0.0))
    return (pred, info)

def geometry(viz):
    """
        Извлекаем признаки расположения полей на изображении
    """
    ys = {f: viz[f]['box'][1] if viz.get(f) else None for f in FIELDS}
    hs = {f: viz[f]['box'][3] if viz.get(f) else None for f in FIELDS}
    diffs = [ys[b] - ys[a] for a, b in zip(FIELDS[:-1], FIELDS[1:]) if a != 'Birth_place_ru' and ys[a] is not None and (ys[b] is not None)]
    pitch = float(np.median(diffs)) if diffs else np.nan
    heights = [hs[f] for f in FIELDS if f not in ('Birth_place_ru', 'Issue_organization_ru') and hs[f]]
    median_height = float(np.median(heights)) if heights else np.nan
    result = {'pitch': pitch, 'bp_extra': np.nan, 'bp_hratio': np.nan, 'org_hratio': np.nan}
    if ys['Birth_place_ru'] is not None and (not np.isnan(pitch)):
        for k in range(1, 5):
            nxt = FIELDS[5 + k]
            if ys[nxt] is not None:
                result['bp_extra'] = (ys[nxt] - ys['Birth_place_ru'] - k * pitch) / pitch
                break
    if hs['Birth_place_ru'] and (not np.isnan(median_height)):
        result['bp_hratio'] = hs['Birth_place_ru'] / median_height
    if hs['Issue_organization_ru'] and (not np.isnan(median_height)):
        result['org_hratio'] = hs['Issue_organization_ru'] / median_height
    return result

def text_feats(x, prefix, txt, res, own_sets):
    """
        Добавляем признаки длины и частотности текстового поля
    """
    tokens = txt.split() if txt else []
    x[prefix + '_len'] = len(txt or '')
    x[prefix + '_ntok'] = len(tokens)
    x[prefix + '_abbr_ns'] = len(re.findall('(?<![А-ЯЁ])[А-ЯЁ]{1,4}\\.[А-ЯЁ]', txt or ''))
    x[prefix + '_end_prep'] = int(bool(tokens) and tokens[-1] in PREP)
    if tokens:
        x[prefix + '_first_f'] = math.log1p(max(0, res.freq(prefix + '_first', tokens[0], own_sets)))
        x[prefix + '_last_f'] = math.log1p(max(0, res.freq(prefix + '_last', tokens[-1], own_sets)))
        x[prefix + '_oov'] = sum((res.freq(prefix + '_word', token, own_sets) <= 0 for token in tokens))
    else:
        x[prefix + '_first_f'] = np.nan
        x[prefix + '_last_f'] = np.nan
        x[prefix + '_oov'] = np.nan

def featurize(d, res, own=None):
    """
        Объединяем VIZ и MRZ, строим признаки и применяем жёсткие правила
    """
    viz, quality = (d['viz'], d['quality'])
    x = {'sharpness': quality['sharpness'], 'doc_px': quality['doc_px'], 'glare': quality['glare']}
    for source in SOURCES:
        x['src_' + source] = int(d['source'] == source)
    own_sets_values = own_sets(own) if own else None
    raw, mins, line_text, line_conf = ({}, {}, {}, {})
    for field in FIELDS:
        field_data = viz.get(field)
        if field_data and field_data.get('lines'):
            txt = norm(' '.join((line['text'] for line in field_data['lines'])))
            chars = [confidence for line in field_data['lines'] for confidence in line['char_conf']]
            raw[field] = txt or None
            mins[field] = mn(chars)
            x[field + '__low'] = sum((c < 0.99 for c in chars))
            x[field + '__nl'] = len(field_data['lines'])
            if len(field_data['lines']) == 1 and len(field_data['lines'][0]['text']) == len(field_data['lines'][0]['char_conf']):
                line_text[field] = field_data['lines'][0]['text']
                line_conf[field] = field_data['lines'][0]['char_conf']
        else:
            raw[field] = None
            mins[field] = 0.0
            x[field + '__low'] = 0
            x[field + '__nl'] = 0
        x[field + '__min'] = mins[field]
    x['n_null'] = sum((value is None for value in raw.values()))
    x['min_conf_viz'] = min(mins.values())
    mrz = parse_mrz(d.get('mrz'))
    x['mrz_present'] = int(mrz is not None)
    for key in ('ok_len', 'ok_num', 'ok_birth', 'ok_tail'):
        x['mrz_' + key] = int(bool(mrz and mrz[key]))
    fused, fmin = ({}, {})
    checks = viz_checks(raw, mrz)
    stat = Counter()
    for field in STRUCT:
        viz_value = raw[field] if raw[field] and FORMAT[field].fullmatch(raw[field]) else None
        hard_value = mrz['hard'].get(field) if mrz else None
        soft_value = mrz['soft'].get(field) if mrz else None
        mrz_conf = mrz['mins'].get(field, 0.0) if mrz else 0.0
        if viz_value and hard_value:
            status = 'agree' if viz_value == hard_value else 'conflict'
            value = hard_value
            confidence = max(mins[field], mrz_conf) if status == 'agree' else mrz_conf
        elif hard_value:
            status, value, confidence = ('mrz_only', hard_value, mrz_conf)
        elif viz_value and soft_value and (viz_value == soft_value):
            status = 'agree_soft'
            value = viz_value
            confidence = max(mins[field], mrz_conf)
        elif viz_value and checks[field]:
            status, value, confidence = ('viz_chk', viz_value, mins[field])
        elif viz_value:
            status, value, confidence = ('viz_only', viz_value, mins[field])
        elif soft_value:
            status, value, confidence = ('soft_only', soft_value, mrz_conf)
        else:
            status, value, confidence = ('none', None, 0.0)
        stat[status] += 1
        fused[field] = value
        fmin[field] = confidence
    for status in STATUSES:
        x['n_' + status] = stat[status]
    name_status = Counter()
    for field, key in NAMES.items():
        viz_value = raw[field]
        value = viz_value
        confidence = mins[field]
        status = 'viz_only' if viz_value else 'none'
        if mrz and mrz['names'] is not None:
            mrz_name = mrz['names'][key]
            mrz_conf = mrz['names_conf'][key]
            trunc = mrz['trunc'] and key == 'middle'
            if viz_value is not None:
                if name_agree(viz_value, mrz_name, trunc):
                    status = 'agree'
                    confidence = max(mins[field], mrz_conf)
                else:
                    repaired = None
                    if field in line_conf and (not trunc):
                        repaired = repair_name(line_text[field], line_conf[field], mrz_name, mrz['names_confl'][key])
                    if repaired:
                        status = 'repaired'
                        value = norm(repaired)
                        confidence = min(mins[field], mrz_conf)
                    else:
                        status = 'mismatch'
                        confidence = 0.0
            elif key == 'middle' and mrz_name == '' and (not mrz['trunc']):
                status = 'empty'
                value = ''
                confidence = mrz_conf
            elif mrz_name and (not trunc) and all((c in INV for c in mrz_name)):
                status = 'mrz_only'
                value = ''.join((INV[c] for c in mrz_name))
                confidence = mrz_conf
        name_status[status] += 1
        fused[field] = value
        fmin[field] = confidence
    x['n_name_mismatch'] = name_status['mismatch']
    x['n_name_repaired'] = name_status['repaired']
    x['n_name_agree'] = name_status['agree']
    x['n_name_mrz_only'] = name_status['mrz_only']
    x['n_name_empty'] = name_status['empty']
    first_name = fused['First_name_ru']
    middle_name = fused['Middle_name_ru']
    x['fn_oov'] = int(res.freq('fn_word', first_name, own_sets_values) <= 0) if first_name else np.nan
    x['mn_oov'] = int(res.freq('mn_word', middle_name, own_sets_values) <= 0) if middle_name else np.nan
    fused['Birth_place_ru'] = raw['Birth_place_ru']
    fmin['Birth_place_ru'] = mins['Birth_place_ru']
    text_feats(x, 'bp', raw['Birth_place_ru'], res, own_sets_values)
    code = fused['Issue_organisation_code']
    candidates = res.org_cands(code, own['Issue_organization_ru'] if own else None) if code else {}
    org, info = snap_org(raw['Issue_organization_ru'], candidates)
    x.update(info)
    fused['Issue_organization_ru'] = org
    fmin['Issue_organization_ru'] = mins['Issue_organization_ru']
    text_feats(x, 'org', org, res, own_sets_values)
    x['code_known'] = int(bool(code) and (code in res.ref or code in res.org))
    g = geometry(viz)
    x.update(g)
    x['bp_extra_resid'] = g['bp_extra'] - 0.62 * (x['Birth_place_ru__nl'] - 1) if not np.isnan(g['bp_extra']) else np.nan
    birth = parse_date(fused['Birth_date'])
    issue = parse_date(fused['Issue_date'])
    valid = passport_valid(birth, issue)
    x['valid'] = -1 if valid is None else int(valid)
    x['issue_before_14'] = int(bool(birth and issue and (issue < add_years(birth, 14))))
    for field in FIELDS:
        x[field + '__fmin'] = fmin[field]
    x['min_conf_fused'] = min(fmin.values())
    x['n_miss'] = sum((fused[field] is None for field in FIELDS))
    x['bp_hdiff'] = abs(g['bp_hratio'] - x['Birth_place_ru__nl']) if not np.isnan(g['bp_hratio']) else np.nan
    x['org_hdiff'] = abs(g['org_hratio'] - x['Issue_organization_ru__nl']) if not np.isnan(g['org_hratio']) else np.nan
    hard_ok = x['n_miss'] == 0 and x['n_name_mismatch'] == 0 and (x['n_name_repaired'] == 0) and (x['valid'] == 1) and (x['issue_before_14'] == 0)
    return (x, fused, hard_ok)

def build_features(preds, res, labels=None, loo=False):
    """
        Строим таблицу признаков и отмечаем документы с ошибками
    """
    rows = []
    fused_list = []
    hard = []
    wrong = []
    ids = []
    sources = []
    for doc_id, d in preds.items():
        label = labels.get(doc_id) if labels else None
        x, fused, ok = featurize(d, res, label['fields'] if label and loo else None)
        rows.append(x)
        fused_list.append(fused)
        hard.append(ok)
        ids.append(doc_id)
        sources.append(d['source'])
        if label:
            expected = label['fields']
            fields_wrong = any((norm(fused[field]) != norm(expected[field]) for field in FIELDS))
            valid = passport_valid(parse_date(expected['Birth_date']), parse_date(expected['Issue_date']))
            wrong.append(fields_wrong or valid is not True)
    X = pd.DataFrame(rows)
    return (X, fused_list, np.array(hard), np.array(wrong) if labels else None, ids, np.array(sources))

def make_model(cols):
    """
        Создаём классификатор вероятности ошибки
    """
    mono = [-1 if col.endswith('__fmin') or col in ('min_conf_fused', 'min_conf_viz') else 0 for col in cols]
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=250, min_samples_leaf=20, l2_regularization=1.0, monotonic_cst=mono, random_state=0)

def cross_val_proba(X, y, reps=2, seed=0):
    """
        Оцениваем вероятность ошибки на частях обучающей выборки
    """
    probabilities = np.zeros(len(y))
    for repeat in range(reps):
        splitter = StratifiedKFold(5, shuffle=True, random_state=seed + repeat)
        for train_idx, valid_idx in splitter.split(X, y):
            model = make_model(X.columns)
            model.fit(X.iloc[train_idx], y[train_idx])
            probabilities[valid_idx] += model.predict_proba(X.iloc[valid_idx])[:, 1]
    return probabilities / reps

def choose_threshold(probabilities, wrong, lam):
    """
        Подбираем строгий порог с большим штрафом за ошибочный приём
    """
    order = np.argsort(probabilities, kind='stable')
    errors = wrong[order].astype(int)
    count = np.arange(1, len(probabilities) + 1)
    utility = count - (1 + lam) * np.cumsum(errors)
    best = utility.max()
    index = int(np.argmax(utility >= best - max(2.0, 0.02 * best)))
    return (float(probabilities[order][index]), index + 1)

def upper(errors, count):
    """
        Оцениваем верхнюю границу числа ошибок с запасом
    """
    return (errors + 2 + 1.96 * math.sqrt(errors + 1)) / max(count, 1)

def main():
    """
        Загружаем данные, обучаем модель и сохраняем решения accept/reject
    """
    parser = argparse.ArgumentParser(description="Постобработка OCR паспортов")
    parser.add_argument("input", help="JSONL с предсказаниями")
    parser.add_argument("output", help="путь для answers.jsonl")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    data_dir = base / "data"
    ref_path = base / "reference" / "subdivisions.csv"

    labels = load_jsonl(data_dir / "train_labels.jsonl")
    train_predictions = load_jsonl(data_dir / "train_predictions.jsonl")
    predictions = load_jsonl(args.input)
    resources = Resources(labels, ref_path)

    X_train, _, hard_train, wrong_train, _, sources_train = build_features(
        train_predictions, resources, labels, loo=True
    )

    # Оставляем для обучения только документы, 
    # прошедшие качественную (жесткую) проверку
    pool = hard_train
    X = X_train[pool].reset_index(drop=True)
    y = wrong_train[pool].astype(int)
    sources = sources_train[pool]

    print(
        f"Обучающая выборка после жёстких правил: {pool.sum()}/{len(pool)}, "
        f"с ошибкой {y.sum()} ({y.mean():.1%})",
        file=sys.stderr,
    )

    check_probabilities = cross_val_proba(X, y, reps=2)
    threshold, _ = choose_threshold(check_probabilities, y.astype(bool), lam=50.0)
    selected = check_probabilities <= threshold
    check_errors = int(y[selected].sum())
    print(
        f"Порог вероятности ошибки <= {threshold:.4f}: принято "
        f"{selected.sum()} из {len(X)} ({selected.mean():.1%}), "
        f"ошибок {check_errors}",
        file=sys.stderr,
    )

    # Обучаем итоговую модель на нашей выборке
    model = make_model(X.columns)
    model.fit(X, y)
    X_test, fused_test, hard_test, _, ids, sources_test = build_features(predictions, resources)

    probabilities = np.ones(len(ids))
    if hard_test.any():
        probabilities[hard_test] = model.predict_proba(
            X_test[hard_test].reindex(columns=X.columns)
        )[:, 1]
    accept = hard_test & (probabilities <= threshold)

    # Записываем только поля принятых документов
    with open(args.output, "w", encoding="utf-8") as fh:
        for i, doc_id in enumerate(ids):
            if accept[i]:
                record = {
                    "doc_id": doc_id,
                    "decision": "accept",
                    "fields": {field: fused_test[i][field] or "" for field in FIELDS},
                }
            else:
                record = {"doc_id": doc_id, "decision": "reject"}
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"\nОбработано документов: {len(ids)}, прошли жёсткие правила "
        f"{int(hard_test.sum())}, принято {int(accept.sum())} ({accept.mean():.1%})"
    )
    expected_errors = upper_errors = 0.0
    print("\nСтатистика по источникам снимков:")
    for source in sorted(set(sources_test)):
        train_selected = selected & (sources == source)
        train_count = int(train_selected.sum())
        train_errors = int(y[train_selected].sum())
        accepted_count = int((accept & (sources_test == source)).sum())
        if train_count:
            expected_errors += accepted_count * train_errors / train_count
            upper_errors += accepted_count * upper(train_errors, train_count)
        print(
            f"  {source:10s} проверка: принято {train_count:4d}, "
            f"ошибок {train_errors}; обработано: принято {accepted_count}"
        )

    print(f"\nПрогноз ошибок среди принятых документов: примерно {expected_errors:.1f}")
    print(f"Грубая верхняя оценка числа ошибок: около {upper_errors:.0f}")
    print("Важно: это прогноз по обучающей выборке, а не фактическое число ошибок.")


if __name__ == '__main__':
    main()
