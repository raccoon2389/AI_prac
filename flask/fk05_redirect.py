from flask import Flask
from flask import redirect

app = Flask(__name__)


@app.route("/")
def index():
    return redirect("www.naver.com")


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8808, debug=True)
