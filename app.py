from flask import Flask, render_template, request
import ollama

app = Flask(__name__)

client = ollama.Client()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    user_input = request.form['message']

    system_prompt = (
        "You are now an expert in maternal health and women's care. "
        "Look at this prompt and provide a supportive suggestion to help "
        "this woman in whatever context she needs. Be kind, practical, and brief. "
        "Do not diagnose medical conditions."
    )

    api_input = system_prompt + "\n\nUser message: " + user_input

    response = client.generate(
        model='qwen2.5:1.5b',
        prompt=api_input
    )

    output = response['response']

    return render_template('result.html', user_input=user_input, output=output)

if __name__ == '__main__':
    app.run(debug=True)
