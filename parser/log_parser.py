'''
Robust hybrid parser for Moniepoint global financial transaction logs with structured output.
Primary parser: Regex engine (fast and format-specific)
Fallback parser: spaCy NLP engine (for unstructured edge cases)
Tracks raw log excel files in `data/raw_log_data/`, extracts logs, and writes structured excel file to `data/parsed_data/` also writes errors for review to "data/MALFORMED_data".
Targeted for Moniepoint and international scalability.
'''

import os
import re
import logging
import pandas as pd
import spacy
from datetime import datetime
from typing import Dict, Optional

# PATH CONFIGURATION
RAW_LOG_DIR = "data/raw_log_data"
PARSED_OUTPUT_DIR = "data/parsed_data"
MALFORMED_DIR = "data/MALFORMED_data"
PROCESSED_TRACKER = "data/.processed_files.txt"

# Ensure required directories exist
os.makedirs(PARSED_OUTPUT_DIR, exist_ok=True)
os.makedirs(MALFORMED_DIR, exist_ok=True)

# LOGGING SETUP
os.makedirs("parser", exist_ok=True)
logging.basicConfig(filename='parser/parsing_errors.log', level=logging.INFO)
logger = logging.getLogger(__name__)

# NLP LOADING
nlp = spacy.load("en_core_web_sm")

# Currency Map (Expanded for Moniepoint & Global Support)
CURRENCY_SYMBOLS = {"₦": "NGN", "$": "USD", "£": "GBP", "€": "EUR"}

# REGEX PATTERNS FOR STRUCTURED LOGS
regex_patterns = [
    # Format: timestamp::userID::txn_type::amount::location::device
    r'^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})::user(?P<user_id>\d+)::(?P<txn_type>[\w-]+)::(?P<amount>[\d.]+)::(?P<location>[\w_]+|None)::(?P<device>.+)$',

    # Format: timestamp | user: userID | txn: top-up of £2074.1 from location | device: device
    r'^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| user: user(?P<user_id>\d+) \| txn: (?P<txn_type>[\w-]+) of (?P<currency>[₦$£€])?(?P<amount>[\d.]+) from (?P<location>[\w_]+|None) \| device: (?P<device>.+)$',

    # Format: timestamp - user=userID - action=txn_type currency amount - ATM: location - device=device
    r'^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - user=user(?P<user_id>\d+) - action=(?P<txn_type>[\w-]+) (?P<currency>[₦$£€])?(?P<amount>[\d.]+) - ATM: (?P<location>\w+|None) - device=(?P<device>.+)$',

    # Format: dd/mm/yyyy hh:mm:ss ::: userID *** txn_type ::: amt:amount currency @ location <device>
    r'^(?P<timestamp>\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}) ::: user(?P<user_id>\d+).*?(?P<txn_type>[\w-]+).*?amt:(?P<amount>[\d.]+)(?P<currency>[₦$£€]) @ (?P<location>.+?) <(?P<device>.+)>$',

    # Format: userID timestamp txn_type amount location device
    r'^user(?P<user_id>\d+)\s+(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(?P<txn_type>[\w-]+)\s+(?P<amount>[\d.]+)\s+(?P<location>\w+|None)\s+(?P<device>.+)$',

    # Format: usr:userID|txn_type|currency amount|location|timestamp|device
    r'^usr:user(?P<user_id>\d+)\|(?P<txn_type>[\w-]+)\|(?P<currency>[₦$£€])(?P<amount>[\d.]+)\|(?P<location>\w+|None)\|(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\|(?P<device>.+)$',

    # Format: timestamp >> [userID] did txn_type - amt=currency amount - location // dev:device
    r'^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) >> \[user(?P<user_id>\d+)\] did (?P<txn_type>[\w-]+) - amt=(?P<currency>[₦$£€])(?P<amount>[\d.]+) - (?P<location>[\w_]+|None) // dev:(?P<device>.+)$',
]

# REGEX PARSER
def parse_log_regex(log: str) -> Dict[str, Optional[str]]:
    for pattern in regex_patterns:
        match = re.match(pattern, log)
        if match:
            return match.groupdict()
    return {}

# NLP FALLBACK PARSER
def parse_log_nlp(log: str) -> Dict[str, Optional[str]]:
    result = {'timestamp': None, 'user_id': None, 'txn_type': None, 'amount': None,
              'currency': None, 'location': None, 'device': None}

    ts = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}|\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}', log)
    if ts:
        raw = ts.group(0)
        fmt = '%d/%m/%Y %H:%M:%S' if '/' in raw else '%Y-%m-%d %H:%M:%S'
        try:
            result['timestamp'] = datetime.strptime(raw, fmt).strftime('%Y-%m-%d %H:%M:%S')
        except:
            pass

    u = re.search(r'user:?\s*\[?user(\d+)\]?|usr:user(\d+)|user(\d+)', log)
    if u:
        result['user_id'] = next((g for g in u.groups() if g), None)

    for t in ['withdrawal', 'deposit', 'debit', 'cashout', 'refund', 'purchase', 'transfer', 'top-up']:
        if t in log.lower():
            result['txn_type'] = t
            break

    doc = nlp(log)
    for ent in doc.ents:
        if ent.label_ == "MONEY":
            am = re.search(r'([₦$£€]?)([\d.]+)', ent.text)
            if am:
                result['currency'] = am.group(1)
                result['amount'] = am.group(2)
        elif ent.label_ == "GPE":
            result['location'] = ent.text

    dev = re.search(r'device[:=]([^|]+)|dev:([^/]+)|<(.+?)>', log)
    if dev:
        result['device'] = next((g for g in dev.groups() if g), None)

    return result

# GENERALIZED PARSER
def parse_log(log: str) -> Dict[str, Optional[str]]:
    result = parse_log_regex(log)
    if not result or sum(v is not None for v in result.values()) < 5:
        result = parse_log_nlp(log)

    return {
        'timestamp': result.get('timestamp'),
        'user_id': result.get('user_id'),
        'txn_type': result.get('txn_type'),
        'amount': result.get('amount'),
        'currency': result.get('currency'),
        'location': result.get('location'),
        'device': result.get('device')
    }

# BULK PARSER
def parse_logs(logs):
    parsed, malformed = [], []
    for log in logs:
        res = parse_log(str(log))
        if sum(v is not None for v in res.values()) >= 6:
            parsed.append(res)
        else:
            malformed.append(log)

    df = pd.DataFrame(parsed)
    if not df.empty and 'timestamp' in df:
        df[['date', 'time']] = df['timestamp'].str.split(' ', expand=True)
    return df[['date', 'time', 'user_id', 'txn_type', 'amount', 'currency', 'device', 'location']], malformed

# PROCESS A SINGLE EXCEL FILE
def process_log_file(filepath: str):
    try:
        df = pd.read_excel(filepath, engine='openpyxl')
        if 'raw_log' not in df.columns:
            logger.warning(f"'raw_log' column missing in {filepath}")
            return

        parsed_df, malformed = parse_logs(df['raw_log'])
        base = os.path.splitext(os.path.basename(filepath))[0]
        parsed_df.to_excel(f"{PARSED_OUTPUT_DIR}/{base}_parsed.xlsx", index=False)

        if malformed:
            pd.DataFrame({'raw_log': malformed}).to_excel(f"{MALFORMED_DIR}/{base}_malformed.xlsx", index=False)

        print(f"Processed: {filepath}")

    except Exception as e:
        logger.error(f"Error processing {filepath}: {e}")
        print(f"Failed to process {filepath}: {e}")

# FILE TRACKER
def load_processed_files():
    return set(open(PROCESSED_TRACKER).read().splitlines()) if os.path.exists(PROCESSED_TRACKER) else set()

def save_processed_file(fname):
    with open(PROCESSED_TRACKER, 'a') as f:
        f.write(fname + '\n')

# WATCH AND PARSE ALL FILES
def watch_and_parse():
    processed = load_processed_files()
    for fname in os.listdir(RAW_LOG_DIR):
        fpath = os.path.join(RAW_LOG_DIR, fname)
        if fname.endswith('.xlsx') and fname not in processed:
            process_log_file(fpath)
            save_processed_file(fname)

if __name__ == "__main__":
    watch_and_parse()