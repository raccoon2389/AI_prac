from flask import Flask,make_response, Response

app = Flask(__name__)


@app.route("/")
def response_test():
    custom_response = Response("★Custom Response",200,{"Program":"Flask Web Application"})
    return make_response(custom_response)

@app.before_first_request
def before_first():
    print('[1]앱이 기동되고 나서 첫번쨰 http 요청에만 응답한다.')



@app.before_request
def before_request():
    print("[2]매 http 요청응이애앤안트ㅏㅌㅊ티ㅣㅣ이이ㅣ잉이잉")

@app.after_request
def after(response):
    print("[3]after")
    return response

@app.teardown_request
def teardown(exception):
    print('[4]브라우저 다음')

@app.teardown_appcontext
def tra(exception):
    print('[5]앱 컨텍스트가 종료될때 실행')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8008, debug=True)


