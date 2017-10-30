from datetime import datetime, timedelta
import csv

from oandapyV20 import API
from oandapyV20.contrib.factories import InstrumentsCandlesFactory

token = '435fc89e4fec6d2c1fa46de985e6ab6f-2b4e3458cb268aac64666f37d10ac0ee'

client = API(access_token=token)

RFC3339_format_accept = "%Y-%m-%dT%H:%M:%SZ"

_from = datetime(2013, 1, 1, 0, 0, 0)
_to = datetime.today()
granularity = 'M15'
instrument = 'EUR_USD'
count_max_per_request = 5000

params = {
    'granularity': granularity,
    'from': _from.strftime(RFC3339_format_accept),
    'to': _to.strftime(RFC3339_format_accept),
}

incomplete = list()

with open('eur_usd_m15.csv', mode='w', newline='') as csvfile:
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
