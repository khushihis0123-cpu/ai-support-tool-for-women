import uuid
from analytics import init_db, log_turn
from flask import Flask, render_template, request, session, redirect
from flask_session import Session
import ollama
import os
import re
import numpy as np
from rag_embeddings import build_index, search, model as embedding_model
from rag import load_knowledge
import markdown as md

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-key")
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = "./.flask_sessions"

Session(app) 
init_db()
client = ollama.Client()

MODEL = "llama3.2:3b"

chunks = load_knowledge()
faiss_index, embeddings, chunks = build_index(chunks)

CONCERN_EMBEDDINGS = {
    "pain": embedding_model.encode(
        "nipple pain soreness cracking bleeding burning during breastfeeding",
        normalize_embeddings=True
    ),
    "latch": embedding_model.encode(
        "baby not latching trouble latching poor latch difficulty attaching to breast",
        normalize_embeddings=True
    ),
    "supply": embedding_model.encode(
        "low milk supply not enough milk baby still hungry after feeding",
        normalize_embeddings=True
    ),
    "stress": embedding_model.encode(
        "overwhelmed exhausted crying giving up failing anxious stressed can't do this",
        normalize_embeddings=True
    ),
    "red_flag": embedding_model.encode(
        "fever mastitis abscess baby not feeding at all dehydration no wet diapers blood severe infection emergency",
        normalize_embeddings=True
    ),
    "concern": embedding_model.encode(
        "worried unsure milk supply questions feeding often cluster feeding reassurance",
        normalize_embeddings=True
    ),
}

THRESHOLDS = {
    "pain": 0.15,
    "latch": 0.15,
    "supply": 0.15,
    "stress": 0.15,
    "red_flag": 0.50, 
    "concern": 0.20,
}

def clean_output(text):
    return md.markdown(text)


def build_conversation_text(chat_history):
    text = ""
    for msg in chat_history:
        text += f"{msg['role'].capitalize()}: {msg['content']}\n"
    return text


def check_hard_medical_red_flags(user_input, chat_history):
    combined_text = user_input.lower()
    for msg in chat_history:
        combined_text += " " + msg["content"].lower()

    # ── NEW: Baby danger signals (fever, feeding refusal) ──────────────────
    infant_danger_words = [
        "fever", "not drinking", "not eating", "won't eat", "won't drink",
        "not feeding", "refusing to feed", "refusing to eat", "hasn't fed",
        "not fed", "won't feed"
    ]
    baby_words = ["baby", "infant", "newborn", "he", "she", "they"]
    if (
        any(d in combined_text for d in infant_danger_words)
        and any(b in combined_text for b in baby_words)
        # make sure it isn't the mother describing her own fever + breast issue
        # (that case is caught by the maternal block below)
        and not (
            any(inf in combined_text for inf in ["mastitis", "streaks", "chills", "flu-like"])
            and any(br in combined_text for br in ["breast", "nipple", "boob"])
        )
    ):
        print("🚨 INFANT DANGER FLAG: Baby fever/feeding refusal detected!", flush=True)
        return "INFANT_URGENT"
    # ── END NEW BLOCK ───────────────────────────────────────────────────────

    diaper_pattern = r"\b(1|2|3|4|one|two|three|four)\b\s*(wet)?\s*diaper"
    feed_pattern = r"\b(1|2|3|4|one|two|three|four)\b\s*(feeds|feedings|times)\s*(a|per)?\s*day"
    
    if re.search(diaper_pattern, combined_text) and "more than" not in combined_text and "at least" not in combined_text:
        return "INFANT_URGENT"
    if re.search(feed_pattern, combined_text):
        return "INFANT_URGENT"

    infection_words = ["fever", "mastitis", "101", "102", "103", "104", "chills", "streaks", "flu-like"]
    breast_words = ["breast", "nipple", "boob"]
    if any(inf in combined_text for inf in infection_words) and any(br in combined_text for br in breast_words):
        print("🚨 HARD MEDICAL FLAG TRIGGERED: Potential systemic breast infection detected!", flush=True)
        return "MATERNAL_URGENT"

    return None


def score_text_concern(user_input, chat_history):
    recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history

    full_context_text = user_input
    for msg in recent_history:
        full_context_text += " " + msg["content"]

    context_vec = embedding_model.encode(
        full_context_text,
        normalize_embeddings=True
    )

    scores = {k: 0 for k in CONCERN_EMBEDDINGS.keys()}

    print("\n--- SEMANTIC SIMILARITY DEBUG ---", flush=True)
    for category, concern_vec in CONCERN_EMBEDDINGS.items():
        similarity = float(np.dot(context_vec, concern_vec))
        print(f"Category: {category:<10} | Score: {similarity:.3f} | Triggered: {similarity >= THRESHOLDS[category]}", flush=True)
        if similarity >= THRESHOLDS[category]:
            scores[category] = 1
    print("---------------------------------\n", flush=True)

    return scores


def detect_detail_level(text):
    text = text.lower()
    detailed = ["months", "weeks", "feeds", "hours", "per day", "twice", "once", "schedule", "old", "days old"]
    vague = ["hungry", "worried", "concerned", "not sure", "help", "dont know", "don't know", "something's wrong"]

    if any(w in text for w in detailed):
        return "DETAILED"
    if any(w in text for w in vague):
        return "VAGUE"
    return "NEUTRAL"


def get_conversation_state(chat_history):
    if not chat_history:
        return "NEW"

    last_user = None
    for msg in reversed(chat_history):
        if msg["role"] == "user":
            last_user = msg["content"].lower()
            break

    if last_user and any(w in last_user for w in ["hungry", "feeding", "milk", "latch", "breast", "nipple"]):
        return "ACTIVE_BREASTFEEDING_THREAD"

    return "OTHER"


def baby_age_known(chat_history, user_input):
    all_text = user_input.lower()
    for msg in chat_history:
        all_text += " " + msg["content"].lower()
    age_words = ["day", "days", "week", "weeks", "month", "months", "old", "newborn", "infant"]
    return any(w in all_text for w in age_words)


def needs_clarification(user_input, chat_history):
    text = user_input.lower()
    concern_words = [
        "enough milk", "hungry", "feeding", "feed", "latch", 
        "milk supply", "not satisfied", "getting enough", "weight gain"
    ]

    concern_detected = any(word in text for word in concern_words)
    if "does" in text or "what is" in text or "why do" in text:
        concern_detected = False

    all_text = text + " " + " ".join([msg["content"].lower() for msg in chat_history])

    age_known = any(w in all_text for w in ["day old", "days old", "week old", "weeks old", "month old", "months old", "newborn"])
    feeding_known = any(w in all_text for w in ["times a day", "feeds a day", "feeding every", "every 2 hours", "every 3 hours"])
    
    diaper_known = False
    if any(phrase in all_text for phrase in ["wet diaper", "wet diapers"]):
        if not any(f"{n}" in all_text for n in ["1", "2", "3", "4", "one", "two", "three", "four"]):
            diaper_known = True

    context_score = sum([age_known, diaper_known, feeding_known])
    return context_score < 2


def is_closing_message(user_input):
    closing_words = [
        "thank you", "thanks", "that helped", "that was helpful", "that's helpful",
        "got it", "okay thanks", "great thanks", "ok thanks", "appreciate it",
        "makes sense", "that makes sense", "perfect", "awesome thanks", "helpful thanks"
    ]
    return any(phrase in user_input.lower() for phrase in closing_words)


def route_request(scores, user_input, chat_history):
    if is_closing_message(user_input):
        return "CLOSING"

    flag_status = check_hard_medical_red_flags(user_input, chat_history)
    if flag_status == "INFANT_URGENT":
        return "URGENT_INFANT"
    elif flag_status == "MATERNAL_URGENT":
        return "URGENT_MATERNAL"

    if needs_clarification(user_input, chat_history):
        return "QUESTION_FIRST"

    if scores["red_flag"]:
        return "URGENT_MATERNAL"

    if scores["stress"]:
        return "SUPPORT"

    if scores["concern"] and not (scores["pain"] or scores["latch"] or scores["supply"]):
        return "REASSURE"

    state = get_conversation_state(chat_history)
    detail = detect_detail_level(user_input)
    follow_up_words = ["yes", "yeah", "ok", "okay", "correct", "right", "yep", "yup"]

    if user_input.strip().lower() in follow_up_words and state == "ACTIVE_BREASTFEEDING_THREAD":
        return "CLINICAL"

    clinical_score = scores["pain"] + scores["latch"] + scores["supply"]
    if clinical_score >= 1:  
        return "CLINICAL"

    if state == "ACTIVE_BREASTFEEDING_THREAD" and detail in ["DETAILED", "NEUTRAL"]:
        return "CLINICAL"

    if detail == "VAGUE":
        return "QUESTION_FIRST"

    return "CLINICAL"


@app.route('/')
def index():
    if "chat_history" not in session:
        session["chat_history"] = []
    return render_template("index.html", chat_history=session["chat_history"])


@app.route("/submit", methods=["POST"])
def submit():
    user_input = request.form.get("user_input", "")
    
    if "chat_history" not in session:
        session["chat_history"] = []
    chat_history = session["chat_history"]

    flag_status = check_hard_medical_red_flags(user_input, chat_history)
    
    if flag_status == "INFANT_URGENT":
        infant_response = """
<h3>⚠️ Seek Immediate Pediatric Care</h3>
<p>Your baby's symptoms — including <strong>fever and/or feeding refusal</strong> — require prompt medical evaluation. Do not wait.</p>
<ul>
    <li><strong>Call your pediatrician or go to the ER now.</strong></li>
    <li>Fever in infants under 3 months is always a medical emergency.</li>
    <li>A baby refusing to feed alongside a fever can indicate serious illness.</li>
    <li>Newborns generally need to feed <strong>8–12 times per 24 hours</strong>. Significantly fewer feeds or wet diapers than expected is also a reason to seek care immediately.</li>
</ul>
"""
        return render_template("result.html", user_input=user_input, response=infant_response)

    elif flag_status == "MATERNAL_URGENT":
        maternal_response = """
<h3>⚠️ Immediate Medical Evaluation Recommended</h3>
<p>Your symptoms strongly point toward a systemic infection like <strong>mastitis</strong>.</p>
<ul>
    <li><strong>Seek Professional Care Immediately:</strong> Contact your provider or visit an urgent care today.</li>
    <li><strong>Breastfeeding Guidance:</strong> Standard guidelines advise continuing to breastfeed or pump frequently on the affected side to clear the blockage unless advised otherwise by a doctor.</li>
</ul>
"""
        return render_template("result.html", user_input=user_input, response=maternal_response)

    scores = score_text_concern(user_input, chat_history)
    route = route_request(scores, user_input, chat_history)
    
    retrieved_context = ""
    if route in ["CLINICAL", "URGENT_INFANT", "URGENT_MATERNAL", "QUESTION_FIRST"]:
        search_results = search(user_input, faiss_index, chunks, k=2)
        if search_results:
            retrieved_context = "\n".join(search_results)

    intent_handling_directive = (
        "USER INTENT HANDLING DIRECTIVE:\n"
        "1. Distinguish between an objective, general educational question (e.g., 'Does latching hurt?') "
        "and a personal symptom disclosure (e.g., 'My nipples hurt when latching').\n"
        "2. If the user's input is a general/educational question, provide a direct, objective medical answer "
        "using the retrieved context first. Do NOT use phrasing that assumes the user is actively suffering from "
        "or experiencing the symptom firsthand (avoid 'I'm sorry you are dealing with this pain').\n"
        "3. If the user explicitly mentions they are personally experiencing a symptom or issue, then adopt a supportive, "
        "empathetic tone and follow intake workflows.\n\n"
    )

    if route == "QUESTION_FIRST":
        system_prompt = (
            "You are an empathetic maternal health assistant. The user has raised a breastfeeding or supply concern, "
            "but key details are missing (such as the baby's precise age or feeding/wet diaper frequencies).\n\n"
            f"{intent_handling_directive}"
            "CRITICAL DIRECTIVE:\n"
            "If the message is an objective question, answer it neutrally. If it is a personal problem missing details, "
            "warmly validate their concern and explicitly ask them to share how old their baby is and how often they feed so you can give them customized context."
        )
    else:
        system_prompt = (
            "You are a helpful maternal health assistant. Answer the user's inquiry based strictly on the provided context.\n"
            f"{intent_handling_directive}"
            f"Grounding Context:\n{retrieved_context}\n\n"
            "Keep formatting clean using simple Markdown bullet points and bold headers."
        )

    history_context = build_conversation_text(chat_history)
    full_prompt = (
        f"### Instruction:\n{system_prompt}\n\n"
        f"### Conversation History:\n{history_context if history_context else 'No prior history.'}\n\n"
        f"### Current User Input:\n{user_input}\n\n"
        f"### Response:\n"
    )

    try:
        response = client.generate(
            model=MODEL, 
            prompt=full_prompt,
            options={
                "num_predict": 350,   
                "temperature": 0.2,   
                "top_k": 20           
            }
        )
        ai_text = response['response']

        try:
            session_id = session.get("session_id", str(uuid.uuid4()))
            session["session_id"] = session_id
            log_turn(
                session_id=session_id,
                turn_number=len(chat_history) // 2 + 1,
                route=route,
                scores={
                    "pain":    float(scores["pain"]),
                    "latch":   float(scores["latch"]),
                    "supply":  float(scores["supply"]),
                    "stress":  float(scores["stress"]),
                    "urgency": float(scores["red_flag"]),
                },
                flags={
                    "pain":    scores["pain"],
                    "latch":   scores["latch"],
                    "supply":  scores["supply"],
                    "stress":  scores["stress"],
                    "urgency": scores["red_flag"],
                },
                detail_level=detect_detail_level(user_input),
                conv_state=get_conversation_state(chat_history),
                retrieval_chunks=[retrieved_context] if retrieved_context else [],
                retrieval_scores=[0.5] if retrieved_context else [],
                user_msg_len=len(user_input),
                baby_age_known=baby_age_known(chat_history, user_input),
                is_closing=is_closing_message(user_input),
            )
        except Exception as log_err:
            print(f"⚠️ Analytics Database Sync Skipped: {log_err}", flush=True)

        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": ai_text})
        session["chat_history"] = chat_history

        html_response = md.markdown(ai_text)
        return render_template("result.html", user_input=user_input, response=html_response)

    except Exception as e:
        print(f"💥 Backend Prompt Generation Error: {e}", flush=True)
        return render_template("result.html", user_input=user_input, response=f"<p>Execution Error: {e}</p>")


@app.route('/reset', methods=['GET', 'POST'])
def reset():
    session.pop("chat_history", None)
    return redirect('/')


if __name__ == '__main__':
    app.run(debug=True, port=5001)
