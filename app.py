from flask import Flask, send_from_directory
from alfworld_play_tw_api import app as alfworld_app

app = alfworld_app

@app.route('/')
def serve_ui():
    return send_from_directory('.', 'static_alfworld_play_tw.html')

if __name__ == '__main__':
    app.run(debug=True) 