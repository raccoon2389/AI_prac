from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello333():
    return "<h1>hello world</h1>"


@app.route('/bit')
def hello334():
    return "<h1>bbbbit</h1>"

if __name__ =='__main__':
    app.run(host='127.0.0.1',port=8888,debug=True)

