import json

from oandapyV20.contrib.requests import MarketOrderRequest, TakeProfitDetails, StopLossDetails
import oandapyV20.endpoints.orders as orders
import oandapyV20

from exampleauth import exampleAuth

access_token = "3282a2c5c3e6e10c06ae7d04f365ae25-efe560b3801a46ee6ad6fe781632a78e"
accountID = "101-004-12211600-001"

api = oandapyV20.API(access_token=access_token)

# EUR_USD (today 1.0750)
EUR_USD_STOP_LOSS = 1.07
EUR_USD_TAKE_PROFIT = 1.10

mktOrder = MarketOrderRequest(
    instrument="EUR_USD",
    units=1,
    takeProfitOnFill=TakeProfitDetails(price=EUR_USD_TAKE_PROFIT).data,
    stopLossOnFill=StopLossDetails(price=EUR_USD_STOP_LOSS).data)

# create the OrderCreate request
r = orders.OrderCreate(accountID, data=mktOrder.data)
try:
    # create the OrderCreate request
    rv = api.request(r)
except oandapyV20.exceptions.V20Error as err:
    print(r.status_code, err)
else:
    print(json.dumps(rv, indent=2))
