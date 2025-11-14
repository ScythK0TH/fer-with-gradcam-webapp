// static/js/main.js
// simple reload mechanism if feed is broken
const img = document.getElementById('video');
img.onerror = function() {
    console.log("Stream error, retrying in 1s...");
    setTimeout(() => { img.src = '/video_feed?' + Date.now(); }, 1000);
};
