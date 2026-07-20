const imageInput = document.querySelector("#imageInput");
const reconstructButton = document.querySelector("#reconstructButton");
const resultMessage = document.querySelector("#resultMessage");
const inputCard = document.querySelector("#inputCard");
const inputCardStatus = document.querySelector("#inputCardStatus");
const outputCard = document.querySelector("#outputCard");
const encoderStage = document.querySelector("#encoderStage");
const decoderStage = document.querySelector("#decoderStage");
const latentVisual = document.querySelector("#latentVisual");
const inputImage = document.querySelector("#inputImage");
const reconstructedImage = document.querySelector("#reconstructedImage");

let selectedImage = null;
let inputPreviewUrl = null;
let processing = false;

const isLocal = ["localhost", "127.0.0.1"].includes(window.location.hostname) || window.location.protocol === "file:";
const apiBase = isLocal && window.location.port !== "8000" ? "http://127.0.0.1:8000" : "";
const apiEndpoint = `${apiBase}/api/reconstruct`;

function delay(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function showImage(imageElement, source) {
  imageElement.src = source;
  imageElement.parentElement.classList.add("has-image");
}

function hideImage(imageElement) {
  imageElement.removeAttribute("src");
  imageElement.parentElement.classList.remove("has-image");
}

function setNetwork(stage, status, active) {
  stage.classList.toggle("is-active", active);
  stage.querySelector(".stage-status").textContent = status;
}

function create32PixelPreview(file) {
  return new Promise((resolve, reject) => {
    const source = new Image();
    const url = URL.createObjectURL(file);
    source.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = 32;
      canvas.height = 32;
      const context = canvas.getContext("2d");
      context.fillStyle = "#000";
      context.fillRect(0, 0, 32, 32);
      context.drawImage(source, 0, 0, 32, 32);
      URL.revokeObjectURL(url);
      resolve(canvas.toDataURL("image/png"));
    };
    source.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("The selected file could not be read as an image."));
    };
    source.src = url;
  });
}

function resetPipeline() {
  inputCard.classList.remove("is-preparing");
  outputCard.classList.remove("is-revealing");
  latentVisual.classList.remove("is-filled");
  setNetwork(encoderStage, "waiting", false);
  setNetwork(decoderStage, "waiting", false);
  hideImage(inputImage);
  hideImage(reconstructedImage);
}

async function setSelectedFile(file) {
  selectedImage = file;
  reconstructButton.disabled = true;
  resetPipeline();
  resultMessage.className = "result-message";
  resultMessage.textContent = "Resizing the selected image to a 32 × 32 preview…";

  try {
    inputPreviewUrl = await create32PixelPreview(file);
    showImage(inputImage, inputPreviewUrl);
    inputCardStatus.textContent = "32 × 32 preview · click to replace";
    reconstructButton.disabled = false;
    resultMessage.textContent = "Image ready. Run the reconstruction when you are ready.";
  } catch (error) {
    selectedImage = null;
    inputCardStatus.textContent = "click the card to upload";
    resultMessage.className = "result-message error";
    resultMessage.textContent = error.message;
  }
}

async function receiveFiles(files) {
  const [file] = files;
  if (!file || processing) return;
  if (!file.type.startsWith("image/")) {
    resultMessage.className = "result-message error";
    resultMessage.textContent = "Please choose a PNG, JPG, or WebP image.";
    return;
  }
  if (file.size > 4 * 1024 * 1024) {
    resultMessage.className = "result-message error";
    resultMessage.textContent = "That image is larger than 4 MB. Try a smaller upload.";
    return;
  }
  await setSelectedFile(file);
}

async function runReconstruction() {
  if (!selectedImage || processing) return;
  processing = true;
  inputCard.disabled = true;
  reconstructButton.disabled = true;
  reconstructButton.firstElementChild.textContent = "Processing…";
  resultMessage.className = "result-message";
  outputCard.classList.remove("is-revealing");
  latentVisual.classList.remove("is-filled");
  setNetwork(encoderStage, "waiting", false);
  setNetwork(decoderStage, "waiting", false);
  hideImage(reconstructedImage);

  const form = new FormData();
  form.append("image", selectedImage);
  const request = fetch(apiEndpoint, { method: "POST", body: form });

  try {
    resultMessage.textContent = "Preparing the 32 × 32 image for the encoder…";
    inputCard.classList.add("is-preparing");
    await delay(1000);
    inputCard.classList.remove("is-preparing");

    resultMessage.textContent = "The encoder is compressing 3,072 colour values into 512 values…";
    setNetwork(encoderStage, "encoding", true);
    await delay(1600);
    setNetwork(encoderStage, "encoded", false);

    resultMessage.textContent = "The 512-value latent space now holds the image's essential signal.";
    latentVisual.classList.add("is-filled");
    await delay(950);

    resultMessage.textContent = "The decoder is rebuilding a new 32 × 32 image from that compact code…";
    setNetwork(decoderStage, "decoding", true);
    await delay(1000);
    setNetwork(decoderStage, "finishing output", true);

    const response = await request;
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "The model could not reconstruct that image.");

    showImage(inputImage, payload.input_32x32);
    showImage(reconstructedImage, payload.reconstruction);
    setNetwork(decoderStage, "decoded", false);
    outputCard.classList.add("is-revealing");
    resultMessage.textContent = `Complete. ${payload.input_values.toLocaleString()} input values became ${payload.latent_values.toLocaleString()} latent values—a ${payload.compression_ratio}× smaller representation.`;
    await delay(900);
    outputCard.classList.remove("is-revealing");
  } catch (error) {
    resultMessage.className = "result-message error";
    resultMessage.textContent = error.message || "Something went wrong. Please try again.";
    setNetwork(encoderStage, "waiting", false);
    setNetwork(decoderStage, "waiting", false);
  } finally {
    processing = false;
    inputCard.disabled = false;
    reconstructButton.disabled = !selectedImage;
    reconstructButton.firstElementChild.textContent = "Run reconstruction";
  }
}

inputCard.addEventListener("click", () => {
  if (processing) return;
  imageInput.value = "";
  imageInput.click();
});
imageInput.addEventListener("change", (event) => receiveFiles(event.target.files));
reconstructButton.addEventListener("click", runReconstruction);

["dragenter", "dragover"].forEach((eventName) => inputCard.addEventListener(eventName, (event) => {
  event.preventDefault();
  if (!processing) inputCard.classList.add("dragging");
}));
["dragleave", "drop"].forEach((eventName) => inputCard.addEventListener(eventName, (event) => {
  event.preventDefault();
  inputCard.classList.remove("dragging");
}));
inputCard.addEventListener("drop", (event) => receiveFiles(event.dataTransfer.files));
