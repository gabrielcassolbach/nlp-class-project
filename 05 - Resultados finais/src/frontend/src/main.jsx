import { useState } from "react";
import { createRoot } from "react-dom/client";
import ChatHistory from "./components/chathistory/chatHistory.jsx";
import ChatInput from "./components/chatinput/chatInput";
import "./app.css";

function App() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  async function sendMessage(text) {
    console.log("SEND MESSAGE")
    if (!text.trim()) return;
    setMessages((prev) => [...prev, `You: ${text}`]);
    setIsLoading(true);

    const res = await fetch("/message", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: text,
          top_k: 3
        }),
    });

    const data = await res.json();
    console.log("data: ", data)

    setMessages((prev) => [...prev, `LLM: ${data.answer}`]);
    setIsLoading(false);
  }

  return (
    <div className="app-root">
      <video
        autoPlay
        loop
        muted
        playsInline
        className="background-video"
      >
        <source src="/static/background.mp4" type="video/mp4" />
        Your browser does not support the video tag.
      </video>

      <div className="chat-container">
        <ChatHistory messages={messages} isLoading={isLoading}/>
      </div>
       <ChatInput onSend={sendMessage} />
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
