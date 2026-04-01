import React from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import "./styles/global.css";

createRoot(document.getElementById("root")).render(
  <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
    <App />
  </BrowserRouter>
);

// Register service worker for offline + push notifications
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/sw.js")
      .then(reg => console.log("ARIA SW registered:", reg.scope))
      .catch(err => console.log("SW registration failed:", err));
  });
}

// Request notification permission proactively
if ("Notification" in window && Notification.permission === "default") {
  document.addEventListener("click", function requestOnce() {
    Notification.requestPermission();
    document.removeEventListener("click", requestOnce);
  }, { once: true });
}
