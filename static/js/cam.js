const video = document.getElementById("cam");
const result = document.getElementById("result");

async function startCamera() {
    let stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
}

const ws = new WebSocket("ws://" + window.location.host + "/ws");

ws.onmessage = (msg) => {
    result.src = "data:image/jpeg;base64," + msg.data;
};

function capture() {
    let canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    let ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    let base64 = canvas.toDataURL("image/jpeg").split(",")[1];
    ws.send(base64);
}

setInterval(capture, 100);  // ส่ง 10 FPS

startCamera();
