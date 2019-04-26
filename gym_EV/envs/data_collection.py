from sklearn.mixture import GaussianMixture
import pymongo
import bson
import numpy as np
from datetime import datetime, date
import pandas as pd
import pytz
from collections import defaultdict


def get_holidays():
    holidays = {date(2018, 5, 28),
                date(2018, 7, 3),
                date(2018, 7, 4),
                date(2018, 7, 5),
                date(2018, 9, 3),
                date(2018, 11, 12),
                date(2018, 11, 22),
                date(2018, 11, 23),
                date(2018, 12, 24),
                date(2018, 12, 25),
                date(2018, 12, 31),
                date(2019, 1, 1)}
    return holidays


def skip(ct, cond):
    if cond == 'WEEKDAY' and (ct.isoweekday() >= 6 or ct.date() in get_holidays()):
        return True
    elif cond == 'WEEKEND' and (ct.isoweekday() < 6 or ct.date() in get_holidays()):
        return True
    elif cond == 'HOLIDAY' and ct.date() not in get_holidays():
        return True
    return False


def get_data(acn, start, end, cond='ALL'):
    host, user, pwd = open('db_auth.txt', 'rb').read().decode().split('\n')
    client = pymongo.MongoClient(host=host, username=user, password=pwd)
    options = bson.codec_options.CodecOptions(tz_aware=True)
    col = client[acn].get_collection('sessions', codec_options=options)

    docs = col.find({'connectionTime': {'$gte': start, '$lt': end}},
                    {'connectionTime': 1, 'disconnectTime': 1, 'kWhDelivered': 1,'_id':0})

    d = []
    for e in docs:
        ct = e['connectionTime'].astimezone()
        dt = e['disconnectTime'].astimezone()
        if skip(ct, cond):
            continue
        start_time = (ct - ct.replace(hour=0, minute=0, second=0, microsecond=0)).seconds / 3600
        sojourn = (dt - ct).seconds / 3600
        d.append([start_time, sojourn, e['kWhDelivered']])
    return np.array(d)


def get_user_data(doc):
    if doc.get('userInputs') is None:
        return np.NaN, np.NaN, np.NaN
    return (int(doc['userInputs'][-1]['userID']),
            doc['userInputs'][-1]['kWhRequested'],
            doc['userInputs'][-1]['minutesAvailable'] / 60)


def get_data_by_user(acn, start, end, cond='ALL'):
    host, user, pwd = open('db_auth.txt', 'rb').read().decode().split('\n')
    client = pymongo.MongoClient(host=host, username=user, password=pwd)
    options = bson.codec_options.CodecOptions(tz_aware=True)
    col = client[acn].get_collection('sessions', codec_options=options)

    docs = col.find({'connectionTime': {'$gte': start, '$lt': end}},
                    {'connectionTime': 1, 'disconnectTime': 1, 'kWhDelivered': 1, 'userInputs': 1,  '_id':0})

    real = defaultdict(list)
    inputs = defaultdict(list)
    for e in docs:
        ct = e['connectionTime'].astimezone()
        dt = e['disconnectTime'].astimezone()
        if skip(ct, cond):
            continue
        start_time = (ct - ct.replace(hour=0, minute=0, second=0, microsecond=0)).seconds / 3600
        sojourn = (dt - ct).seconds / 3600
        user_id, user_kWh_requested, user_duration = get_user_data(e)
        real[user_id].append([start_time, sojourn, e['kWhDelivered']])
        inputs[user_id].append([start_time, user_duration, user_kWh_requested])

    for user in real:
        real[user] = np.array(real[user])
        inputs[user] = np.array(inputs[user])
    return real, inputs


def get_sessions(coll, start, end, tz="America/Los_Angeles", min_kWh=1, cond='ALL'):
    local = pytz.timezone(tz)
    Q_START = local.localize(start).astimezone(local)
    Q_END = local.localize(end).astimezone(local)

    filter_criterion = {'connectionTime': {'$gte': Q_START, '$lte': Q_END}, 'kWhDelivered': {'$gte': min_kWh}}
    projection = {'connectionTime': 1, 'disconnectTime': 1, 'doneChargingTime': 1, 'kWhDelivered': 1, '_id': 0}
    data_cur = coll.find(filter_criterion, projection)

    stats = {}
    for d in data_cur:
        ct = d['connectionTime'].astimezone(local)
        dct = d['doneChargingTime'].astimezone(local) if d.get('doneChargingTime') is not None else np.nan
        dt = d['disconnectTime'].astimezone(local)
        if skip(ct, cond):
            continue
        charging_duration = (dct - ct).total_seconds() / 3600 if d.get('doneChargingTime') is not None else np.nan
        session_duration = (dt - ct).total_seconds() / 3600
        stats[ct] = {'Disconnect Time': dt, 'Done Charging': dct,
                     'Session Duration': session_duration, 'Charging Duration': charging_duration,
                     'kWhDelivered': d['kWhDelivered']}
    return pd.DataFrame(stats).T.infer_objects()