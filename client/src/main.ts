import "./style.css";
import * as api from "./api";
import { Camera, requestOrientationPermissionIfNeeded, watchLevel } from "./camera";

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Элемент #${id} не найден`);
  return el as T;
};

const STORAGE_KEY = "puzzlevision.puzzleId";

const serverStatusEl = $<HTMLSpanElement>("server-status");

// --- этап настройки пазла ---
const setupScreen = $<HTMLElement>("setup-screen");
const setupForm = $<HTMLFormElement>("setup-form");
const puzzleIdInput = $<HTMLInputElement>("puzzle-id-input");
const referenceInput = $<HTMLInputElement>("reference-input");
const gridRowsInput = $<HTMLInputElement>("grid-rows-input");
const gridColsInput = $<HTMLInputElement>("grid-cols-input");
const setupErrorEl = $<HTMLParagraphElement>("setup-error");
const setupExistingEl = $<HTMLParagraphElement>("setup-existing");

// --- навигация ---
const navBar = $<HTMLElement>("nav-bar");
const navCameraBtn = $<HTMLButtonElement>("nav-camera-btn");
const navAssemblyBtn = $<HTMLButtonElement>("nav-assembly-btn");
const navPuzzleNameEl = $<HTMLSpanElement>("nav-puzzle-name");

// --- камера/просмотр ---
const cameraScreen = $<HTMLElement>("camera-screen");
const batchStatusEl = $<HTMLParagraphElement>("batch-status");
const videoEl = $<HTMLVideoElement>("camera-video");
const placeholderEl = $<HTMLDivElement>("camera-placeholder");
const startCameraBtn = $<HTMLButtonElement>("start-camera-btn");
const captureBtn = $<HTMLButtonElement>("capture-btn");
const cameraErrorEl = $<HTMLParagraphElement>("camera-error");
const levelIndicatorEl = $<HTMLDivElement>("level-indicator");
const levelTextEl = $<HTMLSpanElement>("level-text");

const reviewScreen = $<HTMLElement>("review-screen");
const capturedPhotoEl = $<HTMLImageElement>("captured-photo");
const retakeBtn = $<HTMLButtonElement>("retake-btn");
const uploadBtn = $<HTMLButtonElement>("upload-btn");
const uploadStatusEl = $<HTMLParagraphElement>("upload-status");

// --- сборка ---
const assemblyScreen = $<HTMLElement>("assembly-screen");
const assemblySummaryEl = $<HTMLParagraphElement>("assembly-summary");
const refreshStepsBtn = $<HTMLButtonElement>("refresh-steps-btn");
const stepsListEl = $<HTMLDivElement>("steps-list");
const assemblyEmptyEl = $<HTMLParagraphElement>("assembly-empty");

// --- AR (этап 6) ---
const navArBtn = $<HTMLButtonElement>("nav-ar-btn");
const arScreen = $<HTMLElement>("ar-screen");
const arVideoEl = $<HTMLVideoElement>("ar-video");
const arOverlayEl = $<HTMLCanvasElement>("ar-overlay");
const arPlaceholderEl = $<HTMLDivElement>("ar-placeholder");
const arStartCameraBtn = $<HTMLButtonElement>("ar-start-camera-btn");
const arStatusEl = $<HTMLParagraphElement>("ar-status");
const arErrorEl = $<HTMLParagraphElement>("ar-error");

const camera = new Camera(videoEl);
const arCamera = new Camera(arVideoEl);
let lastPhotoBlob: Blob | null = null;
let stopLevelWatch: (() => void) | null = null;
let puzzleId: string | null = localStorage.getItem(STORAGE_KEY);
let arPollHandle: ReturnType<typeof setInterval> | null = null;
let arRequestInFlight = false;

type MainTab = "camera" | "assembly" | "ar";

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

function showSetupScreen(): void {
  setupScreen.hidden = false;
  navBar.hidden = true;
  cameraScreen.hidden = true;
  reviewScreen.hidden = true;
  assemblyScreen.hidden = true;
}

function showMainTab(tab: MainTab): void {
  setupScreen.hidden = true;
  navBar.hidden = false;
  navPuzzleNameEl.textContent = puzzleId ?? "";

  navCameraBtn.classList.toggle("nav-bar__btn--active", tab === "camera");
  navAssemblyBtn.classList.toggle("nav-bar__btn--active", tab === "assembly");
  navArBtn.classList.toggle("nav-bar__btn--active", tab === "ar");

  cameraScreen.hidden = tab !== "camera";
  reviewScreen.hidden = true;
  assemblyScreen.hidden = tab !== "assembly";
  arScreen.hidden = tab !== "ar";

  if (tab === "assembly") void refreshAssembly();
  if (tab !== "ar") stopArPolling();
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
  if (!lastPhotoBlob || !puzzleId) return;
  uploadBtn.disabled = true;
  uploadStatusEl.textContent = "Отправка…";
  try {
    const result = await api.uploadBatch(puzzleId, lastPhotoBlob);
    if (result.accepted) {
      uploadStatusEl.textContent = `Партия ${result.batch_number}: найдено ${result.pieces_found_in_batch} деталей (всего в каталоге: ${result.pieces_total})`;
      batchStatusEl.textContent = `Деталей в каталоге: ${result.pieces_total} · шагов сборки: ${result.steps_total}`;
    } else {
      uploadStatusEl.textContent = `Кадр отбракован: ${result.rejection_reason ?? "причина не указана"} — переснимите`;
    }
    showCameraScreen();
  } catch (err) {
    uploadStatusEl.textContent = "Ошибка отправки. Проверьте соединение с сервером.";
    console.error(err);
  } finally {
    uploadBtn.disabled = false;
  }
}

function stepCard(step: api.AssemblyStep): HTMLElement {
  const el = document.createElement("div");
  el.className = "step-card";

  const title = document.createElement("div");
  title.className = "step-card__title";
  title.textContent = `${step.piece_a} (сторона ${step.side_a}) ↔ ${step.piece_b} (сторона ${step.side_b})`;
  el.appendChild(title);

  const meta = document.createElement("div");
  meta.className = "step-card__meta";
  const sectorLabel = step.sector === "frame" ? "рамка" : step.sector === "textured" ? "текстура" : step.sector === "monochrome" ? "однотон" : "?";
  meta.textContent = `Повернуть ${step.piece_b} на ${step.rotation_deg}° · уверенность ${(step.confidence * 100).toFixed(0)}% · зона: ${sectorLabel}`;
  el.appendChild(meta);

  const actions = document.createElement("div");
  actions.className = "step-card__actions";

  const doneBtn = document.createElement("button");
  doneBtn.className = "btn btn--primary";
  doneBtn.textContent = "Готово";
  doneBtn.addEventListener("click", () => void handleStepFeedback(step.step_number, "done"));

  const rejectBtn = document.createElement("button");
  rejectBtn.className = "btn btn--secondary";
  rejectBtn.textContent = "Не подходит";
  rejectBtn.addEventListener("click", () => void handleStepFeedback(step.step_number, "rejected"));

  actions.appendChild(doneBtn);
  actions.appendChild(rejectBtn);
  el.appendChild(actions);

  return el;
}

async function handleStepFeedback(stepNumber: number, status: "done" | "rejected"): Promise<void> {
  if (!puzzleId) return;
  try {
    await api.sendFeedback(puzzleId, stepNumber, status);
    await refreshAssembly();
  } catch (err) {
    console.error(err);
  }
}

async function refreshAssembly(): Promise<void> {
  if (!puzzleId) return;
  try {
    const [status, steps] = await Promise.all([api.getStatus(puzzleId), api.getSteps(puzzleId, 10)]);
    assemblySummaryEl.textContent = `Деталей: ${status.total_pieces} · привязано к образцу: ${status.located ?? "—"} · шагов ожидает: ${status.steps_pending}/${status.steps_total}`;
    stepsListEl.innerHTML = "";
    if (steps.length === 0) {
      assemblyEmptyEl.hidden = false;
    } else {
      assemblyEmptyEl.hidden = true;
      for (const step of steps) stepsListEl.appendChild(stepCard(step));
    }
  } catch (err) {
    assemblySummaryEl.textContent = "Не удалось загрузить статус пазла.";
    console.error(err);
  }
}

async function startArCamera(): Promise<void> {
  arErrorEl.hidden = true;
  try {
    await arCamera.start();
    arPlaceholderEl.hidden = true;
    startArPolling();
  } catch (err) {
    arErrorEl.hidden = false;
    arErrorEl.textContent =
      "Не удалось включить камеру. Разрешите доступ к камере в браузере и убедитесь, что страница открыта по HTTPS.";
    console.error(err);
  }
}

function startArPolling(intervalMs = 1500): void {
  stopArPolling();
  arPollHandle = setInterval(() => void pollArFrame(), intervalMs);
  void pollArFrame();
}

function stopArPolling(): void {
  if (arPollHandle !== null) {
    clearInterval(arPollHandle);
    arPollHandle = null;
  }
}

async function pollArFrame(): Promise<void> {
  if (!puzzleId || arRequestInFlight) return;
  arRequestInFlight = true;
  try {
    const blob = await arCamera.capturePhoto();
    const result = await api.arFrame(puzzleId, blob);
    drawArOverlay(result);
    arStatusEl.textContent = result.marker_found
      ? `Узнано деталей: ${result.pieces.length}`
      : "Маркер не виден — наведите камеру так, чтобы он попал в кадр";
  } catch (err) {
    arStatusEl.textContent = "Не удалось получить подсказку с сервера.";
    console.error(err);
  } finally {
    arRequestInFlight = false;
  }
}

function drawArOverlay(result: api.ARFrameResult): void {
  const rect = arVideoEl.getBoundingClientRect();
  arOverlayEl.width = rect.width;
  arOverlayEl.height = rect.height;
  const ctx = arOverlayEl.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, arOverlayEl.width, arOverlayEl.height);
  if (result.image_width === 0 || result.image_height === 0) return;

  // Видео растянуто на контейнер через object-fit: cover — переводим
  // координаты сервера (в пикселях снятого кадра) в координаты канваса тем
  // же кроп-масштабом, а не простым делением на исходные размеры.
  const scale = Math.max(arOverlayEl.width / result.image_width, arOverlayEl.height / result.image_height);
  const offsetX = (arOverlayEl.width - result.image_width * scale) / 2;
  const offsetY = (arOverlayEl.height - result.image_height * scale) / 2;

  for (const piece of result.pieces) {
    const x = piece.x_px * scale + offsetX;
    const y = piece.y_px * scale + offsetY;

    ctx.beginPath();
    ctx.arc(x, y, 6, 0, 2 * Math.PI);
    ctx.fillStyle = "rgba(217, 126, 46, 0.9)";
    ctx.fill();

    ctx.font = "13px system-ui, sans-serif";
    ctx.fillStyle = "#ffffff";
    ctx.strokeStyle = "rgba(0, 0, 0, 0.7)";
    ctx.lineWidth = 3;
    const label = `${piece.piece_id} ${piece.rotation_deg}°`;
    ctx.strokeText(label, x + 10, y - 10);
    ctx.fillText(label, x + 10, y - 10);
  }
}

async function handleSetupSubmit(ev: SubmitEvent): Promise<void> {
  ev.preventDefault();
  setupErrorEl.hidden = true;
  const id = puzzleIdInput.value.trim();
  if (!id) return;

  const rows = gridRowsInput.value ? Number(gridRowsInput.value) : null;
  const cols = gridColsInput.value ? Number(gridColsInput.value) : null;
  const referenceFile = referenceInput.files?.[0] ?? null;

  try {
    await api.createPuzzle(id, referenceFile, rows, cols);
    puzzleId = id;
    localStorage.setItem(STORAGE_KEY, id);
    showMainTab("camera");
  } catch (err) {
    setupErrorEl.hidden = false;
    setupErrorEl.textContent = err instanceof Error ? err.message : "Не удалось создать пазл";
  }
}

async function init(): Promise<void> {
  void checkServerHealth();

  if (puzzleId) {
    try {
      await api.getStatus(puzzleId);
      showMainTab("camera");
    } catch {
      setupExistingEl.hidden = false;
      setupExistingEl.textContent = `Сохранённый пазл "${puzzleId}" не найден на сервере — создайте новый.`;
      localStorage.removeItem(STORAGE_KEY);
      puzzleId = null;
      showSetupScreen();
    }
  } else {
    showSetupScreen();
  }
}

setupForm.addEventListener("submit", (ev) => void handleSetupSubmit(ev));
navCameraBtn.addEventListener("click", () => showMainTab("camera"));
navAssemblyBtn.addEventListener("click", () => showMainTab("assembly"));
navArBtn.addEventListener("click", () => showMainTab("ar"));
refreshStepsBtn.addEventListener("click", () => void refreshAssembly());
arStartCameraBtn.addEventListener("click", () => void startArCamera());

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
  stopArPolling();
  arCamera.stop();
});

void init();

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/sw.js").catch((err) => console.warn("SW register failed", err));
  });
}
