develop an weconst inputVideo = document.getElementById('inputVideo');
const cameraBtn = document.getElementById('cameraBtn');
const cameraBtnText = document.getElementById('cameraBtnText');
const captureBtn = document.getElementById('captureBtn');
const fileUpload = document.getElementById('fileUpload');
const videoContainer = document.getElementById('videoContainer');

const statusMessage = document.getElementById('statusMessage');
const statusText = document.getElementById('statusText');
const resultsSection = document.getElementById('resultsSection');

const outputCanvas = document.getElementById('outputCanvas');
const outCtx = outputCanvas.getContext('2d', { willReadFrequently: true });

const colorSwatch = document.getElementById('colorSwatch');
const rgbValue = document.getElementById('rgbValue');
const hsvValue = document.getElementById('hsvValue');

const diagnosisCard = document.getElementById('diagnosisCard');
const diagnosisIcon = document.getElementById('diagnosisIcon');
const diagnosisTitle = document.getElementById('diagnosisTitle');
const diagnosisDesc = document.getElementById('diagnosisDesc');

let stream = null;
let currentImage = null;

// Initialize MediaPipe FaceMesh
const faceMesh = new FaceMesh({
  locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
  }
});

faceMesh.setOptions({
  maxNumFaces: 1,
  refineLandmarks: true, // Need for accurate lip mapping
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});

faceMesh.onResults(onResults);

// Outer Lip Landmarks from FaceMesh
const LIP_LANDMARKS = [
  61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185
];

// Utilities
function showStatus(msg) {
  statusMessage.classList.remove('hidden');
  statusText.textContent = msg;
  resultsSection.classList.add('hidden');
}

function hideStatus() {
  statusMessage.classList.add('hidden');
}

// Upload Handling
fileUpload.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  stopCamera();
  const imgUrl = URL.createObjectURL(file);
  const img = new Image();
  img.onload = async () => {
    showStatus("Extracting facial landmarks...");
    outputCanvas.width = img.width;
    outputCanvas.height = img.height;
    outCtx.drawImage(img, 0, 0);
    currentImage = img;
    await faceMesh.send({ image: img });
  };
  img.src = imgUrl;
});

// Webcam Handling
cameraBtn.addEventListener('click', async () => {
  if (stream) {
    stopCamera();
    return;
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
    inputVideo.srcObject = stream;
    videoContainer.classList.remove('hidden');
    cameraBtnText.textContent = "Close Camera";
    cameraBtn.style.backgroundColor = "#6b7280";
    cameraBtn.style.borderColor = "#6b7280";
    resultsSection.classList.add('hidden');
  } catch (err) {
    alert("Camera access denied or unavailable.");
  }
});

captureBtn.addEventListener('click', async () => {
  if (!stream) return;
  const videoW = inputVideo.videoWidth;
  const videoH = inputVideo.videoHeight;

  // Draw current frame to a temporary canvas (and mirror it as in the preview)
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = videoW;
  tempCanvas.height = videoH;
  const tempCtx = tempCanvas.getContext('2d');
  tempCtx.scale(-1, 1);
  tempCtx.drawImage(inputVideo, -videoW, 0, videoW, videoH);

  // Stop camera stream safely
  stopCamera();

  const imgObj = new Image();
  imgObj.onload = async () => {
    showStatus("Analyzing lip coloration...");
    outputCanvas.width = videoW;
    outputCanvas.height = videoH;
    outCtx.drawImage(imgObj, 0, 0);
    currentImage = imgObj;
    await faceMesh.send({ image: imgObj });
  }
  imgObj.src = tempCanvas.toDataURL('image/jpeg');
});

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }
  videoContainer.classList.add('hidden');
  cameraBtnText.textContent = "Use Webcam";
  cameraBtn.style.backgroundColor = "var(--primary)";
  cameraBtn.style.borderColor = "var(--primary)";
}


// Color Math
function rgbToHsv(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  let max = Math.max(r, g, b), min = Math.min(r, g, b);
  let h, s, v = max;
  let d = max - min;
  s = max === 0 ? 0 : d / max;
  if (max === min) {
    h = 0; // achromatic
  } else {
    switch (max) {
      case r: h = (g - b) / d + (g < b ? 6 : 0); break;
      case g: h = (b - r) / d + 2; break;
      case b: h = (r - g) / d + 4; break;
    }
    h /= 6;
  }
  return [h * 360, s * 100, v * 100];
}


// Main processing logic
function onResults(results) {
  hideStatus();
  resultsSection.classList.remove('hidden');

  if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
    const landmarks = results.multiFaceLandmarks[0];
    const w = outputCanvas.width;
    const h = outputCanvas.height;

    // Draw the original image clean first
    outCtx.clearRect(0, 0, w, h);
    if (currentImage) outCtx.drawImage(currentImage, 0, 0, w, h);

    // Create a path along the lips
    outCtx.beginPath();
    for (let i = 0; i < LIP_LANDMARKS.length; i++) {
      const idx = LIP_LANDMARKS[i];
      const point = landmarks[idx];
      const px = point.x * w;
      const py = point.y * h;
      if (i === 0) outCtx.moveTo(px, py);
      else outCtx.lineTo(px, py);
    }
    outCtx.closePath();

    // 1. Draw outline for UI feedback
    outCtx.lineWidth = 2;
    outCtx.strokeStyle = '#3b82f6';
    outCtx.stroke();
    // Fill slightly transparent to highlight extraction zone
    outCtx.fillStyle = 'rgba(255, 255, 255, 0.2)';
    outCtx.fill();

    // 2. Extract image data to find average color
    // Find bounding box for lips to optimize iteration
    let minX = w, minY = h, maxX = 0, maxY = 0;
    LIP_LANDMARKS.forEach(idx => {
      const pt = landmarks[idx];
      minX = Math.min(minX, pt.x * w);
      minY = Math.min(minY, pt.y * h);
      maxX = Math.max(maxX, pt.x * w);
      maxY = Math.max(maxY, pt.y * h);
    });

    minX = Math.max(0, Math.floor(minX));
    minY = Math.max(0, Math.floor(minY));
    maxX = Math.min(w, Math.ceil(maxX));
    maxY = Math.min(h, Math.ceil(maxY));

    const imgData = outCtx.getImageData(minX, minY, maxX - minX, maxY - minY);
    const data = imgData.data;

    let rSum = 0, gSum = 0, bSum = 0, count = 0;

    // We can use a trick: we draw the image strictly clipped to the path on an offscreen canvas to get precise bounds, 
    // or just average a small central box. For robustness in JS, we will sample the bounding box center.
    // For pure polygon checks we use isPointInPath.

    // Create an invisible canvas for perfect clipping
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = w; tempCanvas.height = h;
    const tempCtx = tempCanvas.getContext('2d', { willReadFrequently: true });
    tempCtx.beginPath();
    for (let i = 0; i < LIP_LANDMARKS.length; i++) {
      const pt = landmarks[LIP_LANDMARKS[i]];
      if (i === 0) tempCtx.moveTo(pt.x * w, pt.y * h); else tempCtx.lineTo(pt.x * w, pt.y * h);
    }
    tempCtx.closePath();
    tempCtx.clip();
    tempCtx.drawImage(currentImage, 0, 0, w, h);

    const clippedData = tempCtx.getImageData(minX, minY, maxX - minX, maxY - minY).data;

    for (let i = 0; i < clippedData.length; i += 4) {
      const a = clippedData[i + 3];
      // Exclude fully transparent pixels (outside the lip boundary)
      // Exclude very dark/black pixels assuming they are shadows or teeth cavities
      const r = clippedData[i];
      const g = clippedData[i + 1];
      const b = clippedData[i + 2];

      if (a > 0 && (r > 30 || g > 30 || b > 30)) {
        rSum += r;
        gSum += g;
        bSum += b;
        count++;
      }
    }

    if (count > 0) {
      const avgR = Math.round(rSum / count);
      const avgG = Math.round(gSum / count);
      const avgB = Math.round(bSum / count);

      const hex = `#${(1 << 24 | avgR << 16 | avgG << 8 | avgB).toString(16).slice(1)}`;
      colorSwatch.style.backgroundColor = hex;
      rgbValue.textContent = `RGB(${avgR}, ${avgG}, ${avgB})`;

      const [hue, sat, val] = rgbToHsv(avgR, avgG, avgB);
      hsvValue.textContent = `HSV(${hue.toFixed(1)}°, ${sat.toFixed(1)}%, ${val.toFixed(1)}%)`;

      // Algorithm: Cyanosis determination based on Hue
      // Typical Pink Lips: Hue = 320 to 360, or 0 to 45
      // Cyanosis (Purple/Blue): Hue = ~200 to ~320
      let isCyanosis = false;
      if (hue >= 180 && hue <= 330) {
        // Broadly covers blue to magenta (purple)
        isCyanosis = true;
      }

      if (isCyanosis) {
        diagnosisCard.className = "diagnosis-card high-risk";
        diagnosisIcon.innerHTML = "⚠️";
        diagnosisTitle.textContent = "High Risk (Cyanosis)";
        diagnosisDesc.textContent = "Detected bluish/purplish tints in the lip region which may denote reduced oxygen supply or heart issues.";
      } else {
        diagnosisCard.className = "diagnosis-card normal";
        diagnosisIcon.innerHTML = "✅";
        diagnosisTitle.textContent = "Normal";
        diagnosisDesc.textContent = "Healthy pink/red lip coloration detected. No visible signs of cyanosis.";
      }

    } else {
      alert("Could not accurately isolate lip pixels. Try another image.");
    }

  } else {
    alert("No face detected in the image! Please upload a clear photo.");
    resultsSection.classList.add('hidden');
  }
}
