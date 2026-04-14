from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    stress = int(request.form['stress'])
    sleep = float(request.form['sleep'])
    workload = int(request.form['workload'])
    mood = int(request.form['mood'])
    caregiving_hours = float(request.form['caregiving_hours'])
    # possibly add another on menstraul health

    # if/then logic
    recommendation = ""

    if stress > 7:
        recommendation = "Try a short breathing exercise."
    elif sleep < 5:
        recommendation = "You may need extra rest. Consider a lighter schedule if possible."
    elif mood <= 3:
        recommendation = "Consider checking in with a support person or taking a short break."
    elif caregiving_hours > 8:
        recommendation = "You’ve had a long caregiving day. Try to schedule recovery time."
    else:
        recommendation = "You seem to be doing okay today. Keep using healthy routines."

    return render_template(
        'result.html',
        stress=stress,
        sleep=sleep,
        workload=workload,
        mood=mood,
        caregiving_hours=caregiving_hours,
        recommendation=recommendation
    )

if __name__ == '__main__':
    app.run(debug=True)
