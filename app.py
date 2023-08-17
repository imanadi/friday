import os
import openai
from flask import Flask, redirect, render_template, request, url_for

from details import name, personalDetails

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

conversation_history = []

@app.route("/", methods=("GET", "POST"))
def index():
    global conversation_history

    if request.method == "POST":
        question = request.form["question"]

        # Adding user message to conversation history
        conversation_history.append({"role": "user", "content": question})

        # Creating a message list for API input
        messages = [{"role": "system", "content": loadInfo()}]
        messages.extend(conversation_history)

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        # Adding AI response to conversation history
        ai_response = response.choices[0].message["content"].strip()
        conversation_history.append({"role": "assistant", "content": ai_response})

        return redirect(url_for("index", result=ai_response))

    result = request.args.get("result")
    # Format conversation history for human readability
    formatted_conversation = []
    for message in conversation_history:
        role = name() if message["role"] == "user" else "Jarvis"
        formatted_conversation.append(f"{role}: {message['content']}")

    return render_template("index.html", result=result, conversation_history=formatted_conversation)


if __name__ == "__main__":
    app.run(debug=True)


def loadInfo():
    return """
    From now on, you are my personal assistant Jarvis.
    You have to answer my queries in less than 25 words considering my personality and preferences. 
    """ + personalDetails()
