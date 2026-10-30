def predict(payload):
    # payload is a dict from MLServer
    x = payload["inputs"][0]["data"]
    return {"outputs": [{"data": [sum(x)]}]}