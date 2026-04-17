function switchTab(tabName, event) {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));

    document.getElementById(tabName).classList.add("active");

    if (event) event.target.classList.add("active");
}

const video = document.getElementById("video");
const emotionText = document.getElementById("emotion");
const emoji = document.getElementById("emoji");
const confidenceText = document.getElementById("confidence");
const historyText = document.getElementById("history");

const emojiMap = {
    "Happy": "😄",
    "Sad": "😢",
    "Angry": "😠",
    "Surprise": "😲",
    "Fear": "😨",
    "Disgust": "🤢",
    "Neutral": "😐",
    "No Face": "❌"
};

navigator.mediaDevices.getUserMedia({ video: true })
.then(stream => video.srcObject = stream);

const canvas = document.createElement("canvas");
const ctx = canvas.getContext("2d");

setInterval(() => {

    if (!video.videoWidth) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    ctx.drawImage(video, 0, 0);

    let data = canvas.toDataURL("image/jpeg");

    fetch("/detect", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ image: data })
    })
    .then(res => res.json())
    .then(data => {

        emotionText.innerText = data.emotion;
        emoji.innerText = emojiMap[data.emotion] || "🙂";
        confidenceText.innerText = "Confidence: " + data.confidence;

        // 🧠 HISTORY DISPLAY
        if (data.history) {
            historyText.innerText = data.history.join(" → ");
        }

        // ✨ emoji animation
        emoji.style.transform = "scale(1.15)";
        setTimeout(() => emoji.style.transform = "scale(1)", 150);

    });

}, 800);

function generateGraph() {
    fetch("/graph")
    .then(() => {
        const graph = document.getElementById("graph");

        graph.src = "/static/graph.png?" + new Date().getTime();

        // 🔥 THIS LINE FIXES IT
        graph.classList.add("show");
    })
    .catch(err => {
        console.error("Graph error:", err);
    });
}