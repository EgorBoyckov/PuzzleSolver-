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

const camera = new Camera(videoEl);
let lastPhotoBlob: Blob | null = null;
let stopLevelWatch: (() => void) | null = null;
let puzzleId: string | null = localStorage.getItem(STORAGE_KEY);

type MainTab = "camera" | "assembly";

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

  cameraScreen.hidden = tab !== "camera";
  reviewScreen.hidden = true;
  assemblyScreen.hidden = tab !== "assembly";

  if (tab === "assembly") void refreshAssembly();
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
refreshStepsBtn.addEventListener("click", () => void refreshAssembly());

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

void init();

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/sw.js").catch((err) => console.warn("SW register failed", err));
  });
}
