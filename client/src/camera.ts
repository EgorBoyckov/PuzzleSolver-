/**
 * Обёртка над getUserMedia/ImageCapture: включает камеру телефона (заднюю,
 * если есть) и умеет снять кадр в максимальном доступном разрешении.
 * Если ImageCapture недоступен (не все браузеры его поддерживают), снимаем
 * кадр через canvas — качество чуть ниже, но работает везде.
 */

// Минимальное объявление типов ImageCapture — в lib.dom.d.ts его пока нет.
interface ImageCaptureLike {
  takePhoto(): Promise<Blob>;
}
declare const ImageCapture: { new (track: MediaStreamTrack): ImageCaptureLike } | undefined;

export class Camera {
  private stream: MediaStream | null = null;
  private imageCapture: ImageCaptureLike | null = null;

  constructor(private readonly videoEl: HTMLVideoElement) {}

  async start(): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 4032 },
        height: { ideal: 3024 },
      },
      audio: false,
    });
    this.videoEl.srcObject = this.stream;
    await this.videoEl.play();

    const [track] = this.stream.getVideoTracks();
    if (typeof ImageCapture !== "undefined" && track) {
      try {
        this.imageCapture = new ImageCapture(track);
      } catch {
        this.imageCapture = null;
      }
    }
  }

  stop(): void {
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.imageCapture = null;
  }

  async capturePhoto(): Promise<Blob> {
    if (this.imageCapture) {
      try {
        return await this.imageCapture.takePhoto();
      } catch {
        // некоторые браузеры регистрируют ImageCapture, но takePhoto падает
        // на части устройств — тихо переходим на canvas-фоллбек ниже.
      }
    }
    return this.captureFromVideoElement();
  }

  private captureFromVideoElement(): Promise<Blob> {
    const canvas = document.createElement("canvas");
    canvas.width = this.videoEl.videoWidth;
    canvas.height = this.videoEl.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas 2D недоступен");
    ctx.drawImage(this.videoEl, 0, 0, canvas.width, canvas.height);
    return new Promise((resolve, reject) => {
      canvas.toBlob(
        (blob) => (blob ? resolve(blob) : reject(new Error("Не удалось получить кадр"))),
        "image/jpeg",
        0.92
      );
    });
  }
}

export interface LevelState {
  tiltDeg: number;
  isLevel: boolean;
}

/** Простой индикатор "телефон держат параллельно столу" по DeviceOrientation. */
export function watchLevel(onUpdate: (state: LevelState) => void, thresholdDeg = 6): () => void {
  const handler = (e: DeviceOrientationEvent) => {
    const beta = e.beta ?? 0; // наклон вперёд/назад
    const gamma = e.gamma ?? 0; // наклон влево/вправо
    const tiltDeg = Math.sqrt(beta * beta + gamma * gamma);
    onUpdate({ tiltDeg, isLevel: tiltDeg < thresholdDeg });
  };
  window.addEventListener("deviceorientation", handler);
  return () => window.removeEventListener("deviceorientation", handler);
}

/** iOS требует явного запроса разрешения на DeviceOrientation по жесту пользователя. */
export async function requestOrientationPermissionIfNeeded(): Promise<void> {
  const DOE = DeviceOrientationEvent as unknown as {
    requestPermission?: () => Promise<"granted" | "denied">;
  };
  if (typeof DOE.requestPermission === "function") {
    await DOE.requestPermission();
  }
}
