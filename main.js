const videoElement = document.getElementById('webcam');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const gestureText = document.getElementById('gesture-text');
const statusOverlay = document.getElementById('connection-status');
const startBtn = document.getElementById('start-btn');
const predictionBox = document.querySelector('.prediction-box');

let ws = null;
let camera = null;
let isPredicting = false;

// Initialize WebSocket connection
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

    ws.onopen = () => {
        statusOverlay.textContent = '🟢 Connected to API';
        statusOverlay.className = 'status-overlay connected';
    };

    ws.onmessage = (event) => {
        const response = JSON.parse(event.data);
        if (response.gesture) {
            gestureText.textContent = response.gesture;
            predictionBox.style.boxShadow = '0 0 30px rgba(0, 206, 201, 0.3) inset';
            setTimeout(() => {
                predictionBox.style.boxShadow = '0 0 20px rgba(108, 92, 231, 0.1) inset';
            }, 300);
        } else if (response.error) {
            console.error("API Error:", response.error);
            if (response.error === "Model not loaded") {
                gestureText.textContent = "Model missing!";
                gestureText.style.color = "#ff7675";
            }
        }
    };

    ws.onclose = () => {
        statusOverlay.textContent = '🔴 Disconnected';
        statusOverlay.className = 'status-overlay error';
        setTimeout(connectWebSocket, 3000); // Reconnect loop
    };

    ws.onerror = (error) => {
        console.error("WebSocket Error:", error);
        ws.close();
    };
}

// Helper: Normalize landmarks relative to wrist (lm 0)
function normalizeLandmarks(landmarks) {
    const baseX = landmarks[0].x;
    const baseY = landmarks[0].y;
    
    const normalized = [];
    landmarks.forEach(lm => {
        normalized.push(lm.x - baseX);
        normalized.push(lm.y - baseY);
    });
    return normalized;
}

// Initialize MediaPipe Hands
const hands = new Hands({
    locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
    }
});

hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});

hands.onResults((results) => {
    // Match canvas dims to video
    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;
    
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    // Draw raw video to canvas
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
    
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        const handLandmarks = results.multiHandLandmarks[0];
        
        // Draw landmarks
        drawConnectors(canvasCtx, handLandmarks, HAND_CONNECTIONS,
                       {color: '#00cec9', lineWidth: 4});
        drawLandmarks(canvasCtx, handLandmarks,
                      {color: '#6c5ce7', lineWidth: 2, radius: 4});
        
        // Extract, normalize and send to backend
        if (ws && ws.readyState === WebSocket.OPEN) {
            const normalizedCoords = normalizeLandmarks(handLandmarks);
            ws.send(JSON.stringify(normalizedCoords));
        }
    } else {
        // If no hand detected
        gestureText.textContent = "No hand";
        gestureText.style.color = "var(--text-muted)";
    }
    canvasCtx.restore();
});

// Setup Camera
startBtn.addEventListener('click', () => {
    if (!camera) {
        camera = new Camera(videoElement, {
            onFrame: async () => {
                await hands.send({image: videoElement});
            },
            width: 1280,
            height: 720
        });
        camera.start();
        startBtn.textContent = 'Stop Camera';
        startBtn.style.background = 'linear-gradient(135deg, #ff7675, #d63031)';
    } else {
        camera.stop();
        camera = null;
        startBtn.textContent = 'Start Camera';
        startBtn.style.background = 'linear-gradient(135deg, var(--primary), var(--secondary))';
        
        // Clear canvas
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        gestureText.textContent = "Waiting...";
    }
});

// Start initialization
connectWebSocket();
