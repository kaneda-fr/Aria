"""Offline detector experiment; never imported by the serving application.

Run on Apollo with prepared optional dependencies and a network namespace.
Scores voice selection for pure FR/EN, requested voice for neutral text, and
fallback for mixed/unsupported text. Does not enforce the request language on
an otherwise monolingual reply, change responses, or invoke TTS/devices.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import pathlib
import random
import re
import resource
import statistics
import time
import unicodedata

TOKEN = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?", re.UNICODE)
FR = set("je tu il elle nous vous ils elles le la les une des du est sont avec pour dans quel quelle peux pouvez pas votre cette ces mon mes ma ton ta ses leur mais donc parce pourquoi comment quand combien quoi encore aussi seulement déjà puis maintenant toujours très rien aucun aucune bonjour merci oui non salut désolé impossible comprends fait plaisir accord marche sûr".split())
EN = set("i you he she we they the a an is are with for in what which can could would not your this these my his her their but because why how when much yet also only already then now always very nothing none hello thanks yes no sorry impossible understand done please sure welcome right all".split())
SHORT = {
    'fr': set('oui non merci bonjour bonsoir salut désolé'.split()),
    'en': set('yes no thanks hello goodbye sorry'.split()),
}


def words(text):
    return TOKEN.findall(unicodedata.normalize('NFC', text).casefold())


def lexical(text):
    ts = words(text)
    # Split clitics for French function words without treating every accent as FR.
    ws = set(part for t in ts for part in re.split("['’]", t))
    fr, en = len(ws & FR), len(ws & EN)
    if fr >= 2 and fr - en >= 2:
        return 'fr', fr
    if en >= 2 and en - fr >= 2:
        return 'en', en
    for lang, vocabulary in SHORT.items():
        if ws and ws <= vocabulary:
            return lang, 3
    return None, 0


def segments(text, windows):
    parts = re.split(r'(?<=[.!?;])\s+|\s+[—–]\s+', text.strip())
    result = [s for s in parts if len(words(s)) >= 3] if len(parts) > 1 else []
    if windows:
        matches = list(TOKEN.finditer(text))
        # Bounded six-word windows, including a final window ending at the tail.
        if len(matches) >= 8:
            starts = list(range(0, len(matches) - 5, 3))
            starts.append(len(matches) - 6)
            for start in starts:
                result.append(text[matches[start].start():matches[start + 5].end()])
    return list(dict.fromkeys(result))[:12]


def neutral_or_short(text, hint, inventory):
    canonical = text.strip().rstrip('.!?').casefold()
    if canonical in inventory:
        return hint
    if any(ch.isalpha() and 'LATIN' not in unicodedata.name(ch, '') for ch in text):
        return 'unsupported'
    if re.fullmatch(r'[A-Za-z0-9_.:/-]+', text.strip()) and any(ch.isdigit() for ch in text):
        return hint
    ts = words(text)
    for lang, vocabulary in SHORT.items():
        if ts and set(ts) <= vocabulary:
            return lang
    # Preserve numbers/units, identifiers, isolated names; do not guess from them.
    if not ts or (len(ts) == 1 and ts[0] not in FR | EN):
        return hint
    if re.fullmatch(r'[\d\s.,:%°+\-/]+(?:[A-Za-z]{1,3})?', text.strip()):
        return hint
    return None


def load_current(path):
    tree = ast.parse(path.read_text())
    names = {'speech_language', '_detect_lang_heuristic'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    namespace = {'re': re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def make_detector(name):
    if name == 'cld2':
        import pycld2
        def classify(text):
            reliable, _, details = pycld2.detect(text, isPlainText=True)
            lang = details[0][1]
            return {'lang': None if lang == 'un' else lang,
                    'confidence': 1.0 if reliable else 0.0, 'margin': 1.0 if reliable else 0.0}
        return classify
    if name == 'langid':
        from langid.langid import LanguageIdentifier, model
        detector = LanguageIdentifier.from_modelstring(model, norm_probs=True)
        def classify(text):
            ranking = detector.rank(text)
            return {'lang': ranking[0][0], 'confidence': float(ranking[0][1]),
                    'margin': float(ranking[0][1] - ranking[1][1])}
        return classify
    from lingua import Language, LanguageDetectorBuilder
    builder = LanguageDetectorBuilder.from_languages(Language.FRENCH, Language.ENGLISH,
        Language.PORTUGUESE, Language.SPANISH, Language.GERMAN, Language.ITALIAN)
    if name == 'lingua_low_all':
        builder = LanguageDetectorBuilder.from_all_languages()
    if name in ('lingua_low', 'lingua_low_all'):
        builder = builder.with_low_accuracy_mode()
    detector = builder.with_preloaded_language_models().build()
    def classify(text):
        ranking = detector.compute_language_confidence_values(text)
        first, second = ranking[:2]
        return {'lang': first.language.iso_code_639_1.name.lower() if first.value else None,
                'confidence': first.value, 'margin': first.value - second.value}
    return classify


def classify_rules(text, hint, inventory, current):
    fast = neutral_or_short(text, hint, inventory)
    if fast:
        return fast
    signs = {lexical(s)[0] for s in segments(text, True)} - {None}
    if len(signs) > 1:
        return 'mixed'
    lang, _ = lexical(text)
    return lang or current(text, hint)


def decide(c, detector, policy, inventory, evidence=None):
    if policy['kind'] != 'raw':
        fast = neutral_or_short(c['text'], c['hint'], inventory)
        if fast:
            return fast
    full = evidence['full'] if evidence is not None else detector(c['text'])
    if policy['kind'] == 'raw':
        return full['lang'] or 'unknown'
    cut, margin = policy['confidence'], policy['margin']
    confident = lambda x: x['lang'] and x['confidence'] >= cut and x['margin'] >= margin
    parts = evidence['parts'] if evidence is not None else [detector(s) for s in segments(c['text'], policy['windows'])]
    if evidence is not None and not policy['windows']:
        parts = evidence['sentences']
    labels = {p['lang'] for p in parts if confident(p)}
    if confident(full):
        labels.add(full['lang'])
    if any(lang not in ('fr', 'en') for lang in labels):
        return 'unsupported'
    if len(labels) > 1:
        return 'mixed'
    if confident(full):
        return full['lang']
    # Strong function-word evidence can rescue uncertain short FR/EN replies.
    lang, _ = lexical(c['text'])
    if lang:
        return lang
    return next(iter(labels)) if len(labels) == 1 else 'unknown'


def conservative(c, detector, policy, inventory, current):
    """Combine the existing voice evidence with low-cost language screening."""
    prediction = decide(c, detector, policy, inventory)
    if prediction == 'unknown' and len(words(c['text'])) <= 4:
        return current(c['text'], c['hint'])
    if prediction not in ('fr', 'en'):
        return prediction
    def strong(text):
        fr_hint = current(text, 'fr')
        en_hint = current(text, 'en')
        return fr_hint if fr_hint == en_hint else None
    primary = strong(c['text']) or prediction
    localized = {strong(s) for s in segments(c['text'], True)} - {None}
    return 'mixed' if len(localized | {primary}) > 1 else primary


def score(cases, predictions):
    totals = dict(n=len(cases), correct=0, false_fallback=0, wrong_voice=0,
                  missed_bad=0, bad=0, valid=0, neutral=0, neutral_errors=0)
    errors = []
    for c, p in zip(cases, predictions):
        expected = c['hint'] if c['expected'] == 'neutral' else c['expected']
        bad = expected in ('mixed', 'unsupported')
        fallback = p not in ('fr', 'en')
        ok = fallback if bad else p == expected
        totals['correct'] += int(ok)
        totals['bad' if bad else 'valid'] += 1
        totals['false_fallback'] += int(not bad and fallback)
        totals['wrong_voice'] += int(not bad and not fallback and p != expected)
        totals['missed_bad'] += int(bad and not fallback)
        if c['expected'] == 'neutral':
            totals['neutral'] += 1
            totals['neutral_errors'] += int(not ok)
        if not ok:
            errors.append({'id': c['id'], 'category': c['category'], 'expected': expected,
                           'prediction': p, 'text': c['text'], 'hint': c['hint']})
    totals['errors'] = errors
    return totals


def rss():
    return int(next(line.split()[1] for line in pathlib.Path('/proc/self/status').read_text().splitlines()
                    if line.startswith('VmRSS:'))) / 1024


def timed(fn, cases, seconds):
    randomizer = random.Random(718)
    samples = []
    start = time.perf_counter()
    cpu = time.process_time()
    rounds = 0
    while time.perf_counter() - start < seconds or rounds < 3:
        order = list(range(len(cases)))
        randomizer.shuffle(order)
        for index in order:
            c = cases[index]
            t = time.perf_counter_ns()
            fn(c)
            samples.append((time.perf_counter_ns() - t) / 1e6)
        rounds += 1
    wall = time.perf_counter() - start
    cpu = time.process_time() - cpu
    samples.sort()
    percentile = lambda q: samples[min(len(samples) - 1, math.ceil(len(samples) * q) - 1)]
    return {'samples': len(samples), 'rounds': rounds, 'wall_s': wall, 'cpu_s': cpu,
            'p50_ms': percentile(.5), 'p95_ms': percentile(.95), 'p99_ms': percentile(.99),
            'max_ms': max(samples)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', type=pathlib.Path, required=True)
    ap.add_argument('--source', type=pathlib.Path, required=True)
    ap.add_argument('--detector', choices=['current','rules','cld2','langid','lingua_low','lingua_high','lingua_low_all'], required=True)
    ap.add_argument('--output', type=pathlib.Path, required=True)
    ap.add_argument('--seconds', type=float, default=1.0)
    args = ap.parse_args()
    import numpy  # Same baseline as previous screening; already in Aria's runtime.
    baseline = rss()
    corpus = json.loads(args.corpus.read_text())
    cases = corpus['cases']
    dev = [c for c in cases if c['split'] == 'dev']
    test = [c for c in cases if c['split'] == 'test']
    confirmation = [c for c in cases if c['split'] == 'confirmation']
    fresh = [c for c in cases if c['split'] == 'fresh']
    assert not ({c['text'] for c in dev} & {c['text'] for c in test})
    assert len({c['id'] for c in cases}) == len(cases)
    inventory = {n.casefold() for n in corpus['inventory_names']}
    current = load_current(args.source)
    result = {'detector': args.detector, 'corpus_sha256': hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(), 'baseline_rss_mib': baseline,
        'load_before': pathlib.Path('/proc/loadavg').read_text().strip(), 'profiles': {}}
    if args.detector in ('current','rules'):
        if args.detector == 'current':
            fn = lambda c: current['speech_language'](c['text'], c['hint'])
        else:
            fn = lambda c: classify_rules(c['text'], c['hint'], inventory, current['speech_language'])
        policies = [(args.detector, {}, fn)]
        result['init_ms'] = 0
    else:
        t = time.perf_counter()
        detector = make_detector(args.detector)
        result['init_ms'] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        detector('Bonjour, comment puis-je vous aider ?')
        result['first_call_ms'] = (time.perf_counter() - t) * 1000
        # Only development evidence is available to threshold selection.
        evidence = {c['id']: {'full': detector(c['text']),
                     'parts': [detector(s) for s in segments(c['text'], True)],
                     'sentences': [detector(s) for s in segments(c['text'], False)]} for c in dev}
        candidates = []
        for windows in (False, True):
            for cut in (.5, .7, .85, .95):
                for margin in (.1, .25, .4):
                    policy = dict(kind='guard', windows=windows, confidence=cut, margin=margin)
                    scores = score(dev, [decide(c, detector, policy, inventory, evidence[c['id']]) for c in dev])
                    # Valid replies and unsafe language both matter; false fallback
                    # gets double weight because needless refusal breaks short exchanges.
                    loss = 2 * scores['false_fallback'] + scores['wrong_voice'] + scores['missed_bad']
                    candidates.append((loss, scores['false_fallback'], scores['wrong_voice'], windows, cut, margin, policy))
        chosen = min(candidates, key=lambda item: item[:6])[-1]
        result['selected_on_dev'] = chosen
        result['selection_grid'] = [dict(loss=x[0],false_fallback=x[1],wrong_voice=x[2],**x[-1]) for x in candidates]
        raw = {'kind':'raw'}
        policies = [('raw',raw,lambda c: decide(c,detector,raw,inventory)),
                    ('guard',chosen,lambda c: decide(c,detector,chosen,inventory))]
    if args.detector == 'lingua_low':
        policies.append(('conservative', chosen, lambda c: conservative(c, detector, chosen, inventory, current['speech_language'])))
    for name, policy, fn in policies:
        # Exactly one evaluation after selection; never feed test labels to tuning.
        predictions = [fn(c) for c in cases]
        profile = {'policy':policy,'dev':score(dev,[p for c,p in zip(cases,predictions) if c['split']=='dev']),
                   'test':score(test,[p for c,p in zip(cases,predictions) if c['split']=='test']),
                   'predictions':[dict(id=c['id'],prediction=p) for c,p in zip(cases,predictions)]}
        profile['confirmation'] = score(confirmation, [p for c,p in zip(cases,predictions) if c['split']=='confirmation'])
        if 'detector_warm_rss_mib' not in result:
            result['detector_warm_rss_mib'] = rss()
            result['detector_peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        profile['fresh'] = score(fresh, [p for c,p in zip(cases,predictions) if c['split']=='fresh'])
        profile['timing'] = timed(fn, test + confirmation + fresh, args.seconds)
        result['profiles'][name] = profile
    result.update(warm_rss_mib=rss(),peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                  load_after=pathlib.Path('/proc/loadavg').read_text().strip())
    if args.detector=='current':
        mono=[c for c in cases if c['expected'] in ('fr','en')]
        result['input_heuristic_secondary']=score(mono,[current['_detect_lang_heuristic'](c['text']) for c in mono])
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'detector':args.detector,'selected':result.get('selected_on_dev'),
        'profiles':{n:{'test':{k:v for k,v in p['test'].items() if k!='errors'},'timing':p['timing']} for n,p in result['profiles'].items()},
        'rss_mib':result['detector_warm_rss_mib']-baseline}),flush=True)


if __name__ == '__main__':
    main()
