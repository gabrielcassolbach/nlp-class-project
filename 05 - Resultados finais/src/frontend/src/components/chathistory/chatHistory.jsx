import "./chatHistory.css";

export default function ChatHistory({ messages,  isLoading }) {
  if (!messages.length) return null;

  return (
    <div className="chat-history-container">
      {messages.map((msg, i) => (
        <div
          key={i}
          className={`chat-history-message ${
            msg.startsWith("You:") ? "user" : "bot"
          }`}
        >
          {msg.replace(/^You: |^LLM: /, "")}
        </div>
      ))}

      {isLoading && (
        <div className="chat-history-loading">
          <img src="/static/loading.gif" alt="loading..." />
        </div>
      )}
    </div>
  );
}
