import { useState } from "react";
import UploadPdfButton from "../fileupload/uploadPdfButton";
import "./chatInput.css";

export default function ChatInput({ onSend }) {
  const [text, setText] = useState("");

  function send(t) {
    if (!t.trim()) return;
    setText("");
    onSend(t);
  }

  return (
    <div className="chat-input-container">
      <div className="chat-input-wrapper">
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send(text);
            }
          }}
          placeholder={"Type a message..."}
        />
        <button onClick={() => send(text)}>Send</button>
      </div>
    </div>
  );
}
