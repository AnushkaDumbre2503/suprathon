let mediaRecorder;
let recordedChunks = [];
let isRecorded = false;
let stream;

// Start Recording
async function startRecording() {
  if (isRecorded) {
    alert("You can record only once!");
    return;
  }

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert("Camera/Mic API not supported by this browser.");
    return;
  }

  try {
    // Request camera + microphone access
    stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: true,
    });
    console.log("Camera and microphone accessed successfully!");

    // Show live video preview
    const videoPreview = document.getElementById("videoPreview");
    if (videoPreview) {
      videoPreview.srcObject = stream;
      await videoPreview.play();
    }

    // Initialize MediaRecorder
    mediaRecorder = new MediaRecorder(stream);
    recordedChunks = [];

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) recordedChunks.push(e.data);
    };

    mediaRecorder.onstop = uploadVideo;
    mediaRecorder.start();

    // Stop after 30 seconds
    setTimeout(() => {
      if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
        stopStream();
        isRecorded = true;
      }
    }, 30000);
  } catch (err) {
    console.error("Error accessing camera/microphone:", err.name, err.message);
    alert(`Camera/Mic Error: ${err.name} - ${err.message}`);
  }
}

// Stop all video tracks
function stopStream() {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
  }
}

// Upload recorded video to server
function uploadVideo() {
  const blob = new Blob(recordedChunks, { type: "video/webm" });
  const formData = new FormData();
  formData.append("video", blob);
  const postField = document.getElementById("post");
  formData.append("post", postField ? postField.value : "");

  fetch("/upload", { method: "POST", body: formData })
    .then((res) => res.json())
    .then((data) => alert("Video submitted!"))
    .catch((err) => console.error("Upload error:", err));
}
