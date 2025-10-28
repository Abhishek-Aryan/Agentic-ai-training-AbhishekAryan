

## ✨ SparkBot: Your Personal Creative Companion!

**Welcome to SparkBot**—the AI that's here to banish boring replies forever! Built on the powerful n8n framework and fueled by the Google Gemini Chat Model, SparkBot is designed to be your witty, imaginative, and context-aware chat partner. Think less robot, and more highly-creative digital friend who remembers your past chats and adapts to your mood!

---

### ⚙️ What's Under the Hood? (Workflow Summary)

This flow is super neat and focused on giving you a rich conversational experience.

#### 🧩 The Core Flow, Explained Simply

1.  **The Knock:** You send a chat message, and the **`When chat message received`** node catches it.
2.  **The Brain:** Your message is zipped over to the **`AI Agent`** node—the boss of the operation!
3.  **The Magic:** The agent taps into the **`Google Gemini Chat Model`** for creative firepower and the **`Simple Memory`** node to recall your last few lines.
4.  **The Output:** The agent crafts a unique response, and it shows up in the **`Output (manual)`** node for you to see!

---

### 🧠 The Components That Bring SparkBot to Life

| Component | What it is | Why it's Cool |
| :--- | :--- | :--- |
| **🗨️ When Chat Message Received** | The **Trigger** | It's the ear of the operation! It's always listening for your text input to kick off the fun. (P.S. You can easily swap this for Telegram or Discord later!) |
| **🤖 AI Agent** | The **Logic Unit** | This is the conductor of the orchestra. It ties the model and memory together, making sure the replies are witty, formatted beautifully (using **Markdown**), and full of personality! |
| **🧠 Google Gemini Chat Model** | The **Creative Engine** | This is why SparkBot is awesome! **Gemini excels** at open-ended, expressive chats, letting SparkBot be truly imaginative and context-aware instead of just factual. |
| **💾 Simple Memory** | The **Context Keeper** | It holds onto the last few messages. This is the secret sauce that makes your conversations feel **natural** and connected, not disjointed. |
| **📤 Output (Manual)** | The **Result Viewer** | Right now, it just displays the chat reply for testing. When you're ready, this is where you'll connect a live chat app! |

---

### 💡 Example Scenarios (Go On, Try These!)

SparkBot isn't just for facts—it's for fun, brainstorming, and great conversation:

* **💬 Chat Companion:** Ask it about the universe or the perfect sandwich. **Example:** *"The universe hums a silent symphony, and starlight is its music sheet. 🌌"*
* **🎨 Writing Assistant:** Need a creative boost? **Example:** *"Give me a poetic line about rain."*
* **🧘‍♂️ Mood Companion:** Having a rough day? Spark will cheer you up! **Example:** *"Even clouds have silver linings — let's find yours ☁️✨."*
* **🤓 Educational Twist:** Ask it to explain a tough topic in a silly way. **Example:** *"It's Earth's way of saying, 'Don't float away, buddy!'"*

---

### 🚀 Get SparkBot Running!

Setting this up is a breeze:

1.  **Import** the workflow into your n8n instance.
2.  **Plug In** your Google Gemini API credentials into the Chat Model node (Gotta give Spark its power!).
3.  **Test It!** Click the "Test" button on the `When chat message received` node, type your message, and check the beautiful output in the `Output (manual)` node.
4.  **(Optional but Recommended):** Swap the trigger and output nodes to connect SparkBot to a live platform like Telegram or Discord!

