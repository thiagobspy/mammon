from datetime import datetime, timedelta
import csv

from oandapyV20 import API
from oandapyV20.contrib.factories import InstrumentsCandlesFactory

token = 'f3c6c368f9be6df9a3727506168fb0cf-1afeb072f2da3a861a33af7e41b0d6e0'

client = API(access_token=token)

RFC3339_format_accept = "%Y-%m-%dT%H:%M:%SZ"

_from = datetime(2013, 1, 1, 0, 0, 0)
_to = datetime.today()
granularity = 'M5'
instrument = 'EUR_USD'
count_max_per_request = 5000

params = {
    'granularity': granularity,
    'from': _from.strftime(RFC3339_format_accept),
    'to': _to.strftime(RFC3339_format_accept),
}

incomplete = list()

with open('eur_usd_m5.csv', mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['datetime', 'open', 'high', 'low', 'close', 'volume'])

    for req in InstrumentsCandlesFactory(instrument=instrument, params=params):
        print("REQUEST: {} {} {}".format(req, req.__class__.__name__, req.params))
        response = client.request(req)

        for candle in response.get('candles'):
            try:
                complete = candle['complete']
                if not complete:
                    incomplete.append(candle)
                    continue

                ctime = candle['time'][0:19]
                o = candle['mid']['o']
                h = candle['mid']['h']
                l = candle['mid']['l']
                c = candle['mid']['c']
                v = candle['volume']


            except Exception as e:
                print(e)
            else:
                writer.writerow([ctime, o, h, l, c, v])
