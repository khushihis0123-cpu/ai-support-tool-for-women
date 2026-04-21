from flask import Flask, render_template, request, session
import ollama

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

client = ollama.Client()


def score_text_concern(user_text):
    text = user_text.lower()

    scores = {
        "pain": 0,
        "latch": 0,
        "supply": 0,
        "stress": 0,
        "urgency": 0
    }

    pain_words = [
        "pain", "painful", "hurt", "hurts", "sore", "cracked",
        "bleeding", "burning", "nipple pain", "engorged", "engorgement"
    ]

    latch_words = [
        "latch", "won't latch", "wont latch", "unlatching",
        "shallow latch", "bad latch", "trouble latching", "not latching"
    ]

    supply_words = [
        "low supply", "not enough milk", "still hungry",
        "not satisfied", "not producing enough", "milk supply",
        "baby still hungry", "not getting enough milk"
    ]

    stress_words = [
        "exhausted", "overwhelmed", "frustrated",
        "anxious", "crying", "stressed", "tired"
    ]

    urgent_words = [
        "bleeding", "fever", "baby not feeding",
        "not feeding", "severe pain", "extreme pain"
    ]

    for word in pain_words:
        if word in text:
            scores["pain"] += 1

    for word in latch_words:
        if word in text:
            scores["latch"] += 1

    for word in supply_words:
        if word in text:
            scores["supply"] += 1

    for word in stress_words:
        if word in text:
            scores["stress"] += 1

    for word in urgent_words:
        if word in text:
            scores["urgency"] += 2

    return scores


def get_detected_issues(scores):
    issues = []

    if scores["pain"] > 0:
        issues.append("pain or discomfort during breastfeeding")
    if scores["latch"] > 0:
        issues.append("latch difficulty")
    if scores["supply"] > 0:
        issues.append("milk supply or feeding effectiveness concern")
    if scores["stress"] > 0:
        issues.append("maternal stress, fatigue, or emotional strain")
    if scores["urgency"] > 0:
        issues.append("possible higher-priority concern")

    if not issues:
        issues.append("general breastfeeding concern")

    return issues


@app.route('/')
def index():
    if "chat_history" not in session:
        session["chat_history"] = []
    return render_template("index.html", chat_history=session["chat_history"])


@app.route('/submit', methods=['POST'])
def submit():
    user_input = request.form['message']

    if "chat_history" not in session:
        session["chat_history"] = []

    scores = score_text_concern(user_input)
    detected_issues = get_detected_issues(scores)

    system_prompt = (
        "You are a supportive assistant focused on maternal breastfeeding and postpartum wellness. "
        "Be kind, practical, and brief. "
        "Do not diagnose medical conditions. "
        "Use the prior conversation to answer follow-up questions naturally. "
        "If something sounds more serious, gently suggest contacting a lactation consultant or healthcare provider."
    )

    conversation_text = ""
    for msg in session["chat_history"]:
        conversation_text += f"{msg['role'].capitalize()}: {msg['content']}\n"

    api_input = (
        system_prompt
        + "\n\nConversation so far:\n" + conversation_text
        + "\nUser: " + user_input
        + "\nDetected concern areas for the latest user message: " + ", ".join(detected_issues)
        + "\nRespond helpfully in about 4 to 6 sentences."
    )

    response = client.generate(
        model='qwen2.5:1.5b',
        prompt=api_input
    )

    output = response['response']

    chat_history = session["chat_history"]
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": output})
    session["chat_history"] = chat_history

    return render_template("index.html", chat_history=session["chat_history"])


@app.route('/reset', methods=['POST'])
def reset():
    session.pop("chat_history", None)
    return render_template("index.html", chat_history=[])
    

if __name__ == '__main__':
    app.run(debug=True)
