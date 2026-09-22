import "./style.css";
import { Camera, requestOrientationPermissionIfNeeded, watchLevel } from "./camera";

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Элемент #${id} не найден`);
  return el as T;
};

const serverStatusEl = $<HTMLSpanElement>("server-status");
const videoEl = $<HTMLVideoElement>("camera-video");
const placeholderEl = $<HTMLDivElement>("camera-placeholder");
const startCameraBtn = $<HTMLButtonElement>("start-camera-btn");
const captureBtn = $<HTMLButtonElement>("capture-btn");
const cameraErrorEl = $<HTMLParagraphElement>("camera-error");
const levelIndicatorEl = $<HTMLDivElement>("level-indicator");
const levelTextEl = $<HTMLSpanElement>("level-text");

const cameraScreen = $<HTMLElement>("camera-screen");
const reviewScreen = $<HTMLElement>("review-screen");
const capturedPhotoEl = $<HTMLImageElement>("captured-photo");
const retakeBtn = $<HTMLButtonElement>("retake-btn");
const uploadBtn = $<HTMLButtonElement>("upload-btn");
const uploadStatusEl = $<HTMLParagraphElement>("upload-status");

const camera = new Camera(videoEl);
let lastPhotoBlob: Blob | null = null;
let stopLevelWatch: (() => void) | null = null;

async function checkServerHealth(): Promise<void> {
  try {
    const res = await fetch("/api/health");
    if (!res.ok) throw new Error(String(res.status));
    const body = await res.json();
    serverStatusEl.textContent = `сервер: ${body.version}`;
    serverStatusEl.className = "badge badge--ok";
  } catch {
    serverStatusEl.textContent = "сервер недоступен";
    serverStatusEl.className = "badge badge--error";
  }
}

async function startCamera(): Promise<void> {
  cameraErrorEl.hidden = true;
  try {
    await requestOrientationPermissionIfNeeded();
    await camera.start();
    placeholderEl.hidden = true;
    captureBtn.hidden = false;
    levelIndicatorEl.hidden = false;

    stopLevelWatch = watchLevel((state) => {
      levelTextEl.textContent = state.isLevel
        ? "уровень: ровно ✓"
        : `уровень: наклон ${state.tiltDeg.toFixed(0)}°`;
      levelIndicatorEl.classList.toggle("level-indicator--good", state.isLevel);
    });
  } catch (err) {
    cameraErrorEl.hidden = false;
    cameraErrorEl.textContent =
      "Не удалось включить камеру. Разрешите доступ к камере в браузере и убедитесь, что страница открыта по HTTPS.";
    console.error(err);
  }
}

function showReviewScreen(blob: Blob): void {
  lastPhotoBlob = blob;
  capturedPhotoEl.src = URL.createObjectURL(blob);
  cameraScreen.hidden = true;
  reviewScreen.hidden = false;
  uploadStatusEl.textContent = "";
}

function showCameraScreen(): void {
  reviewScreen.hidden = true;
  cameraScreen.hidden = false;
}

async function uploadPhoto(): Promise<void> {
  if (!lastPhotoBlob) return;
  uploadBtn.disabled = true;
  uploadStatusEl.textContent = "Отправка…";
  try {
    const form = new FormData();
    form.append("file", lastPhotoBlob, `frame_${Date.now()}.jpg`);
    const res = await fetch("/api/debug/upload", { method: "POST", body: form });
    if (!res.ok) throw new Error(String(res.status));
    const body = await res.json();
    uploadStatusEl.textContent = `Сохранено на сервере: ${body.size_bytes} байт`;
  } catch (err) {
    uploadStatusEl.textContent = "Ошибка отправки. Проверьте соединение с сервером.";
    console.error(err);
  } finally {
    uploadBtn.disabled = false;
  }
}

startCameraBtn.addEventListener("click", () => void startCamera());

captureBtn.addEventListener("click", async () => {
  try {
    const blob = await camera.capturePhoto();
    showReviewScreen(blob);
  } catch (err) {
    cameraErrorEl.hidden = false;
    cameraErrorEl.textContent = "Не удалось снять кадр, попробуйте ещё раз.";
    console.error(err);
  }
});

retakeBtn.addEventListener("click", () => {
  showCameraScreen();
});

uploadBtn.addEventListener("click", () => void uploadPhoto());

window.addEventListener("beforeunload", () => {
  stopLevelWatch?.();
  camera.stop();
});

void checkServerHealth();

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/sw.js").catch((err) => console.warn("SW register failed", err));
  });
}
