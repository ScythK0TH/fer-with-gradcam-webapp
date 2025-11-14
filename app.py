# app.py
from flask import Flask, render_template
from flask_sock import Sock

from controllers.ws_stream import create_client_processor

app = Flask(__name__)
sock = Sock(app)

@app.route("/")
def index():
    return render_template("index.html")

@sock.route('/ws')
def ws_route(ws):

    # --- Create private Processor for this Client ---
    processor = create_client_processor()

    while True:
        data = ws.receive()

        if data is None:
            break

        result = processor(data)
        ws.send(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
