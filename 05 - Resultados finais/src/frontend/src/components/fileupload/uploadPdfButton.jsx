import { useRef, useState } from "react";
import "./uploadPdfButton.css";

export default function UploadPdfButton({ onLoadingChange, uploadUrl = "/api/upload" }) {
  const fileInputRef = useRef(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);

  function openPicker() {
    setError(null);
    fileInputRef.current?.click();
  }

  function onFileChange(e) {
    const file = e.target.files?.[0];
    e.target.value = ""; 
    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".pdf") || file.type !== "application/pdf") {
      setError("Only PDF files are supported.");
      return;
    }

    uploadFile(file);
  }

  function uploadFile(file) {
    const xhr = new XMLHttpRequest();
    const form = new FormData();
    form.append("file", file);

    xhr.open("POST", uploadUrl, true);

    xhr.upload.onprogress = (ev) => {
      if (ev.lengthComputable) {
        const pct = Math.round((ev.loaded / ev.total) * 100);
        setProgress(pct);
      }
    };

    xhr.onloadstart = () => {
      setUploading(true);
      setProgress(0);
      setError(null);
      if (typeof onLoadingChange === "function") onLoadingChange(true);
    };

    xhr.onerror = () => {
      setError("Upload failed. Check your connection or server.");
    };

    xhr.onload = () => {
      try {
        const resp = JSON.parse(xhr.responseText || "{}");
        console.log("Debug: ", resp)
        console.log("xhr.status: ", xhr.status)
        if (xhr.status != 200 && !resp.error) {
          setError(resp.error ?? `Upload failed.`);
        }
      } catch (err) {
        setError("Unexpected server response.");
      }
    };

    xhr.onloadend = () => {
      setUploading(false);
      setProgress(0);
      if (typeof onLoadingChange === "function") onLoadingChange(false);
    };

    xhr.send(form);
  }

  return (
    <div className="upload-pdf-root">
      <button
        className="upload-pdf-btn"
        onClick={openPicker}
        disabled={uploading}
        title="Upload PDF"
        aria-label="Upload PDF"
      >
        <span className="upload-plus">＋</span>
      </button>

      <input
        ref={fileInputRef}
        type="file"
        accept="application/pdf"
        style={{ display: "none" }}
        onChange={onFileChange}
      />

      {uploading && (
        <div className="upload-progress" role="progressbar" aria-valuenow={progress} aria-valuemin={0} aria-valuemax={100}>
          <div className="upload-progress-bar" style={{ width: `${progress}%` }} />
          <div className="upload-progress-text">{progress}%</div>
        </div>
      )}

      {error && <div className="upload-error" role="alert">{error}</div>}
    </div>
  );
}
